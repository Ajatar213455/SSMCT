humanact12_kinematic_chain = [[0, 1, 4, 7, 10],
                              [0, 2, 5, 8, 11],
                              [0, 3, 6, 9, 12, 15],
                              [9, 13, 16, 18, 20, 22],
                              [9, 14, 17, 19, 21, 23]]

parents = [0 for i in range(24)]
for chain in humanact12_kinematic_chain:
    for i in range(1, len(chain)):
        parents[chain[i]] = chain[i-1]

print(parents)