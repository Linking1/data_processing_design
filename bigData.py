import numpy as np
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import pairwise_distances
S = dok_matrix((19835, 624961), dtype=np.float32)
S_index = dok_matrix((19835, 624961), dtype=np.float32)
S_norm = dok_matrix((19835, 624961), dtype=np.float32)
S_sim = dok_matrix((19835, 19835), dtype=np.float32)
S_test = dok_matrix((19835, 19835), dtype=np.float32)

fo = open("./input/train.txt", "r")
m = 0
# 19835
while m < 19835:
    m += 1
    test = fo.readline()
    print(test)
    test = test.split('|')
    i = int(test[0])
    k = int(test[1])
    S_index[i, 0] = k
    j = 0
    sum = 0
    mylist = []
    while j < k:
        test = fo.readline()
        test = test.split()
        S[i, int(test[0])] = int(test[1])
        S_index[i, j+1] = int(test[0])
        mylist.append(int(test[0]))
        sum += int(test[1])
        j += 1
    aver = sum/k
    print(aver)
    for aver_i in range(k):
        S_norm[i, mylist[aver_i]] = S[i, mylist[aver_i]] - aver
fo.close()
# item_sim = pairwise_distances(S.T, metric='cosine')
# print(S[1, :])
# print(S_norm[1, :])
# test1 = 550452
# test2 = 323933
# test3 = 159248
# test4 = 554099
# test5 = 70896
# test6 = 518385
# testList = []

fo = open("./input/test.txt", "r")
# 19834
for i in range(19835):
    test = fo.readline()
    print(test)
    test = test.split('|')
    j = int(test[0])
    k = int(test[1])
    for indexj in range(6):
        test = fo.readline()
        S_test[j, indexj] = int(test)
fo.close()

for i in range(19835):
    for j in range(i+1, 19835):
        if S[j, S_test[i, 0]] != 0 or S[j, S_test[i, 1]] != 0 or S[j, S_test[i, 2]] != 0 or S[j, S_test[i, 3]] != 0 or S[j, S_test[i, 4]] != 0 or S[j, S_test[i, 5]] != 0:
            simUp = 0
            simD1 = 0
            simD2 = 0
            for indexj in range(int(S_index[i, 0])):
                simUp += S_norm[i, S_index[i, indexj+1]] * S_norm[j, S_index[i, indexj+1]]
                simD1 += pow(S_norm[i, S_index[i, indexj+1]], 2)
            for indexj in range(int(S_index[j, 0])):
                simD2 += pow(S_norm[j, S_index[j, indexj+1]], 2)
            print("simUp: ", simUp)
            print("simD1: ", simD1)
            print("simD2: ", simD2)
            sim = simUp/(np.sqrt(simD1)*np.sqrt(simD1))
# sim = np.dot(S_norm.toarray()[0, :], S_norm.toarray()[i, :])/(np.linalg.norm(S_norm.toarray()[0, :])*np.linalg.norm(S_norm.toarray()[i, :]))
            print("sim: ", sim)
            print("userId: ", i)
            print("userId: ", j)

            if sim > 0:
                S_sim[i, j] = sim

fo = open("./input/out.txt", "w")
for i in range(19835):
    for j in range(6):
        testNum = 0
        testSim = 0
        for o in range(19835):
            if S[o, S_test[i, j]] != 0:
                if S_sim[i, o] != 0:
                    testNum += S[o, S_test[i, j]] * S_sim[i, o]
                    testSim += S_sim[i, o]
                elif S_sim[o, i] != 0:
                    testNum += S[o, S_test[i, j]] * S_sim[o, i]
                    testSim += S_sim[o, i]
        S[i, S_test[i, j]] = testNum/testSim
        fo.write(str(int(S[i, S_test[i, j]])))
        fo.write('\n')
        print('userid: ', i, 'itemid: ', S_test[i, j], 'sim: ', S[i, S_test[i, j]])
fo.close()
