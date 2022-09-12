import numpy as np
selected_motions = {'kick', 'jump', 'dance', 'walk', 'jog', 'run'}
a = np.array(['dance', 'help'])
b = np.array(['F@Q','QQQ'])
# print("kick" in selected_motions)
print(set(a) & selected_motions)
# print(a.any(a in selected_motions))
# print(parents)