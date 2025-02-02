# Transformer编码器+ 混合嵌入 模型

""" Define the Seq2Seq Generation Network """
import numpy as np
import torch
import torch.nn as nn
try: from layer import MultiHeadAttention, PositionwiseFeedForward
except: from SSMCT.layer import MultiHeadAttention, PositionwiseFeedForward

def get_non_pad_mask(seq):
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(2).ne(0).type(torch.float)
    return non_pad_mask.unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = torch.abs(seq_k).sum(2).eq(0)  # sum the vector of last dim and then judge
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, sliding_windown_size):
    """ For masking out the subsequent info. """
    batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8)
    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    mask = 1 - mask
    # print(mask)
    return mask  # .bool()

def get_keyframe_encoding_table(n_position, d_hid, n_past =10, n_future = 10, n_trans = 30):
    return 0

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table.clone().detach()  # detach 新的tensor开辟与旧的tensor共享内存，新的tensor会脱离计算图，不会牵扯梯度计算

class KeyframeEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200, n_past=10, n_future=10, kf_enc_value = 10):
        super(KeyframeEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('kf_table', self._get_keyframe_table(n_position, d_hid, n_past, n_future, kf_enc_value))

    def _get_keyframe_table(self, n_position, d_hid, n_past, n_future, kf_enc_value):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        kf_table = np.zeros((n_position, d_hid))
        kf_table[n_past:-n_future, :]=kf_enc_value

        return torch.FloatTensor(kf_table).unsqueeze(0)

    def forward(self, x):
        return x + self.kf_table.clone().detach()

class Embedding(nn.Module):
    def __init__(self, device, maxlen=50, d_model=256, n_segments=3):
        super(Embedding, self).__init__()
        self.pos_embed = nn.Embedding(maxlen, d_model).to(device)  # position embedding
        self.kf_embed = nn.Embedding(n_segments, d_model).to(device)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, n_past =10, n_future = 10, n_trans = 30):
        # x  [B, F, d_model]
        seq_len = x.size(1)
        n_position = n_past+n_future+n_trans
        pos_index = torch.arange(seq_len, dtype=torch.long)
        pos_index = pos_index.to(self.device)
        # pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        kf_index = torch.zeros(seq_len, dtype=torch.long)
        kf_index = kf_index.to(self.device)
        kf_index[n_past:n_past+n_trans] = 1
        if(n_position < seq_len):
            kf_index[n_position:] = 2 # 2 for GT encoder
        embedding = x + self.pos_embed(pos_index) + self.kf_embed(kf_index)
        return self.norm(embedding)


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 多头自注意力机制层
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)       # FFN层

    def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """
    An encoder model with self attention mechanism.
    具有自注意力机制的编码器模型
    """
    def __init__(
            self, device, seq_len=50, input_dim=217,      # seq_len：帧数  每个数据的帧数，一个batch中所有数据均为固定长度帧数，如50帧，25帧，35帧  input_dim：输入模型的数据中每一帧的维度
            n_layers=8, n_head=8, d_k=64, d_v=64,   # n_layers：编码器层数  n_head：多头注意力头数  d_k:注意力中矩阵Q，K的维度 d_v:注意力中矩阵V的维度
            d_model=256, d_inner=512, dropout=0.1,  # d_model：卷积层的输出维度  d_inner：FFN隐藏层维度
            n_past =10, n_future = 10, n_trans = 30):   # 已知的关键帧数量和未知帧数

        super().__init__()

        self.name = 'transformer'
        self.d_model = d_model
        self.n_past = n_past
        self.n_future = n_future
        self.n_trans = n_trans

        # 输入卷积层
        self.inConv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding = 1)  # position-wise
        # 混合嵌入层
        self.embedding = Embedding(maxlen=seq_len, d_model=d_model, n_segments=3, device = device)
        # self.position_enc = PositionalEncoding(d_model, n_position=seq_len)
        # self.kf_enc = KeyframeEncoding(d_hid=d_model, n_position=seq_len, n_past=n_past, n_future=n_future, kf_enc_value = kf_enc_value)
        # self.position_encoding = nn.Embedding.from_pretrained(   # 位置编码
        #     get_sinusoid_encoding_table(seq_len, d_model),
        #     freeze=True)

        self.layer_stack = nn.ModuleList([      # 编码器层
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # 输出卷积层
        self.outConv = nn.Conv1d(in_channels=d_model, out_channels=input_dim, kernel_size=3, padding=1)  # position-wise

    def forward(self, src_seq, mask=None, return_attns=False, n_past=None, n_future=None, n_trans=None):
        '''
        :param src_seq: 原始序列，初始为输入的插值序列[B, F, dim]
        :param mask:
        :param return_attns:
        :return:
        '''
        if n_past is None: n_past = self.n_past
        if n_future is None: n_future = self.n_future
        if n_trans is None: n_trans = self.n_trans

        enc_slf_attn_list = []
        # -- Forward
        # 输入卷积层
        src_seq = src_seq.permute(0, 2, 1)  # 注意卷积前后要转维度
        enc_input = self.inConv(src_seq)    # 输入卷积层 [B, F, d_model]
        enc_input = enc_input.permute(0, 2, 1)
        # 混合嵌入
        enc_output = self.embedding(enc_input, n_past=n_past, n_future=n_future, n_trans=n_trans)
        # 位置编码
        # enc_output = self.position_enc(enc_input)   # [B, F, d_model]
        # enc_output = enc_input + self.position_enc(src_seq) # 输入卷积层+位置编码
        # 关键帧编码
        # keyframe_encoding = torch.zeros((enc_output.shape[1], enc_output.shape[2]))
        # keyframe_encoding = torch.zeros_like(enc_output)
        # keyframe_encoding[:, self.n_past:-self.n_future, :] = 1
        # enc_output = enc_output + keyframe_encoding
        # enc_output = self.kf_enc(enc_output)

        # EncoderLayer层
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]   # 注意力分数
        # 输出卷积层
        enc_output = enc_output.permute(0, 2, 1)  # 注意卷积前后要转维度
        enc_output = self.outConv(enc_output)    # 输入卷积层 [B, F, input_dim]
        enc_output = enc_output.permute(0, 2, 1)
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Encoder_BiRNN(nn.Module):
    """
    An encoder model with BiRNN mechanism
    """
    def __init__(
            self, device, seq_len=60, input_dim=168,      # seq_len：帧数  每个数据的帧数，一个batch中所有数据均为固定长度帧数，如50帧，25帧，35帧  input_dim：输入模型的数据中每一帧的维度
            n_layer=None, n_head=None, d_k=None, d_v=None,
            d_model=256, d_inner=256, dropout=0.1,  # d_model：卷积层的输出维度  d_inner：FFN隐藏层维度
            n_past = 15, n_future = 15, n_trans = 30, num_layers = 2):   # 已知的关键帧数量和未知帧数

        super().__init__()

        self.name = 'birnn'
        self.device = device
        self.d_model = d_model
        self.n_past = n_past
        self.n_future = n_future
        self.n_trans = n_trans

        # 输入卷积层
        self.inConv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding = 1)  # position-wise
        # 混合嵌入层
        self.embedding = Embedding(maxlen=seq_len, d_model=d_model, n_segments=3, device = device)

        self.hidden_size = d_model
        self.num_layers = num_layers
        self.lstm = nn.LSTM(d_model, d_inner, num_layers, batch_first=True, bidirectional=True) #input: batch*seq*feature
        self.fc = nn.Linear(d_inner*2, d_model)  # 2 for bidirection

        # 输出卷积层
        self.outConv = nn.Conv1d(in_channels=d_model, out_channels=input_dim, kernel_size=3, padding=1)  # position-wise

    def forward(self, src_seq, mask=None, return_attns=False, n_past=None, n_future=None, n_trans=None):
        '''
        :param src_seq: 原始序列，初始为输入的插值序列[B, F, dim]
        :param mask:
        :param return_attns:
        :return:
        '''
        if n_past is None: n_past = self.n_past
        if n_future is None: n_future = self.n_future
        if n_trans is None: n_trans = self.n_trans

        enc_slf_attn_list = []
        # -- Forward
        # 输入卷积层
        src_seq = src_seq.permute(0, 2, 1)  # 注意卷积前后要转维度
        enc_input = self.inConv(src_seq)    # 输入卷积层 [B, d_model, F]
        enc_input = enc_input.permute(0, 2, 1)
        # 混合嵌入
        enc_output = self.embedding(enc_input, n_past=n_past, n_future=n_future, n_trans=n_trans)
 
        # EncoderLayer层
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, enc_output.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, enc_output.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        # enc_output = enc_output.permute(0, 2, 1) #B*T*C
        enc_output, _ = self.lstm(enc_output, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        B, T, C = enc_output.shape
        enc_output = self.fc(enc_output.reshape(-1, C)).reshape(B, T, -1)
        # enc_output = enc_output.permute(0, 2, 1) #B*C*T

        # 输出卷积层
        enc_output = enc_output.permute(0, 2, 1)  # 注意卷积前后要转维度
        enc_output = self.outConv(enc_output)    # 输入卷积层 [B, input_dim, F]
        enc_output = enc_output.permute(0, 2, 1)

        return enc_output

class CVAE_Transformer_Encoder(nn.Module):
    def __init__(
            self, device, seq_len=50, input_dim=217,      # seq_len：帧数  每个数据的帧数，一个batch中所有数据均为固定长度帧数，如50帧，25帧，35帧  input_dim：输入模型的数据中每一帧的维度
            n_layers=8, n_head=8, d_k=64, d_v=64,   # n_layers：编码器层数  n_head：多头注意力头数  d_k:注意力中矩阵Q，K的维度 d_v:注意力中矩阵V的维度
            d_model=256, d_inner=512, dropout=0.1,  # d_model：卷积层的输出维度  d_inner：FFN隐藏层维度
            n_past =10, n_future = 10, n_trans = 30):   # 已知的关键帧数量和未知帧数

        super().__init__()

        self.d_model = d_model
        self.n_past = 0
        self.n_future = 0
        self.n_trans = 0

        # 输入卷积层
        self.inConv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding = 1)  # position-wise
        # 混合嵌入层
        self.embedding = Embedding(maxlen=seq_len, d_model=d_model, n_segments=3, device = device)

        self.layer_stack = nn.ModuleList([      # 编码器层
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.mu_layer = nn.Linear(d_model, d_model) # 一层Linear够吗？
        self.sigma_layer = nn.Linear(d_model, d_model)

    def forward(self, src_seq, mask=None, return_attns=False):
        '''
        :param src_seq: 原始序列，初始为输入的插值序列[B, F, dim]
        :param mask:
        :param return_attns:
        :return: mu, logvar
        '''
        n_past = self.n_past
        n_future = self.n_future
        n_trans = self.n_trans

        enc_slf_attn_list = []
        # -- Forward
        # 输入卷积层
        src_seq = src_seq.permute(0, 2, 1)  # 注意卷积前后要转维度
        enc_input = self.inConv(src_seq)    # 输入卷积层 [B, d_model, F]
        enc_input = enc_input.permute(0, 2, 1)
        # 混合嵌入
        enc_output = self.embedding(enc_input, n_past=n_past, n_future=n_future, n_trans=n_trans)

        # EncoderLayer层
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]   # 注意力分数
        
        z = enc_output.mean(dim = 1) # calc mean along T axis
        mu = self.mu_layer(z)
        logvar = self.sigma_layer(z)

        return mu, logvar


class CVAE_Transformer_Decoder(nn.Module):
    def __init__(
            self, device, seq_len=50, input_dim=217,      # seq_len：帧数  每个数据的帧数，一个batch中所有数据均为固定长度帧数，如50帧，25帧，35帧  input_dim：输入模型的数据中每一帧的维度
            n_layers=8, n_head=8, d_k=64, d_v=64,   # n_layers：编码器层数  n_head：多头注意力头数  d_k:注意力中矩阵Q，K的维度 d_v:注意力中矩阵V的维度
            d_model=256, d_inner=512, dropout=0.1,  # d_model：卷积层的输出维度  d_inner：FFN隐藏层维度
            n_past =10, n_future = 10, n_trans = 30):   # 已知的关键帧数量和未知帧数

        super().__init__()

        self.d_model = d_model
        self.n_past = n_past
        self.n_future = n_future
        self.n_trans = n_trans

        # 输入卷积层
        self.inConv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding = 1)  # position-wise
        # 混合嵌入层
        self.embedding = Embedding(maxlen=seq_len, d_model=d_model, n_segments=3, device = device)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=d_model,
                                                          nhead=n_head,
                                                          dim_feedforward=d_inner,
                                                          dropout=dropout)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=n_layers)
        # 输出卷积层
        self.outConv = nn.Conv1d(in_channels=d_model, out_channels=input_dim, kernel_size=3, padding=1)  # position-wise

    def forward(self, src_seq, z, mask=None, return_attns=False, n_past=None, n_future=None, n_trans=None):
        '''
        :param src_seq: 原始序列，初始为输入的插值序列[B, F, dim]
        :param z: latent variable
        :param mask:
        :param return_attns:
        :return: new motion
        '''
        if n_past is None: n_past = self.n_past
        if n_future is None: n_future = self.n_future
        if n_trans is None: n_trans = self.n_trans

        # 输入卷积层
        src_seq = src_seq.permute(0, 2, 1)  # 注意卷积前后要转维度
        enc_input = self.inConv(src_seq)    # 输入卷积层 [B, d_model, F]
        enc_input = enc_input.permute(0, 2, 1)
        # 混合嵌入
        enc_output = self.embedding(enc_input, n_past=n_past, n_future=n_future, n_trans=n_trans)
        enc_output = enc_output.permute(1, 0, 2) #BFC->FBC
       
        z = z[None, :, :]
        enc_output = self.seqTransDecoder(tgt=enc_output, memory=z)
        
        enc_output = enc_output.permute(1, 2, 0) # FBC -> BCF
        enc_output = self.outConv(enc_output)
        enc_output = enc_output.permute(0, 2, 1) # BCF -> BFC

        return enc_output


class CVAE_Transformer(nn.Module):
    def __init__(
            self, device, seq_len=50, input_dim=217,      # seq_len：帧数  每个数据的帧数，一个batch中所有数据均为固定长度帧数，如50帧，25帧，35帧  input_dim：输入模型的数据中每一帧的维度
            n_layers=8, n_head=8, d_k=64, d_v=64,   # n_layers：编码器层数  n_head：多头注意力头数  d_k:注意力中矩阵Q，K的维度 d_v:注意力中矩阵V的维度
            d_model=256, d_inner=512, dropout=0.1,  # d_model：卷积层的输出维度  d_inner：FFN隐藏层维度
            n_past =10, n_future = 10, n_trans = 30):   # 已知的关键帧数量和未知帧数
        super().__init__()
        
        self.name = 'CVAE'
        self.device = device
        self.d_model = d_model
        self.encoder = CVAE_Transformer_Encoder(device=device, seq_len=seq_len, input_dim=input_dim, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                                d_model=d_model, d_inner=d_inner, dropout=dropout, n_past=n_past, n_future=n_future, n_trans=n_trans)
        self.decoder = CVAE_Transformer_Decoder(device=device, seq_len=seq_len, input_dim=input_dim, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                                d_model=d_model, d_inner=d_inner, dropout=dropout, n_past=n_past, n_future=n_future, n_trans=n_trans)
    
    def reparameterize(self, mu, logvar, seed=None):
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def compute_kl_loss(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss
    
    def forward(self, batch, mask=None, return_attns=False, n_past=None, n_future=None, n_trans=None):
        '''
        batch = {'gt':  , 'interpolated': ,}
        '''
        mu, logvar = self.encoder(batch['gt'])
        z = self.reparameterize(mu, logvar)
        enc_output = self.decoder(batch['interpolated'], z, n_past=n_past, n_future=n_future, n_trans=n_trans)

        return enc_output, self.compute_kl_loss(mu, logvar)
    
    def generate(self, batch, mask=None, return_attns=False, n_past=None, n_future=None, n_trans=None):
        '''
        batch = {'gt':  , 'interpolated': ,}
        '''
        z = torch.randn(batch['interpolated'].shape[0], self.d_model, device=self.device)
        enc_output = self.decoder(batch['interpolated'], z, n_past=n_past, n_future=n_future, n_trans=n_trans)

        return enc_output