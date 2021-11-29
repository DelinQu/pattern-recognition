#!/usr/bin/env python
# coding: utf-8

import numpy as np

# S,T集合
setS = ('D','N','P','V')
setT = ('a','c')

# 训练集
trainingData = ("NVc","NVc","NVc","NVc","NVc","NVc","NVc","NVc","NVc",
                "PVa","PVa","PVa","PVa","PVa","PVa","PVa","PVa","PVa",
                "NDa")

# 构造目标特征序列
objSeqList = ["NVa","NDa","NPa","NNa"]

# 获取 ydot
def getYdots(t):
    ydots = []
    for s1 in setS:
        for s2 in setS:
            ydot = s1 + s2 + t
            if ydot not in trainingData:
                ydots.append(ydot)
    return ydots

# 获取 fai
def getFai(objSeq,seq):
    # 加上 start and end，分别标定为0,1
    objSeq = '0' + objSeq + '1'
    seq = '0' + seq + '1'
    fai = np.zeros((len(objSeq)-1),dtype=np.int16)
    for i in range(len(objSeq)-1):
        for j in range(len(seq)-1):
            if objSeq[i] == seq[i] and objSeq[i+1] == seq[i+1]:
                fai[i] = fai[i] + 1
    return fai

# 获取梯度 don't observe 部分之和
def getSum(xn,W,objSeq):
    ydots = getYdots(xn)
    expSum, ZSum = 0, 0
    for seq in ydots:
        expP = np.exp(W @ getFai(objSeq,seq))
        ZSum, expSum= ZSum + expP, expSum + expP*getFai(objSeq,seq)
    # 防止 division by zero
    if ZSum == 0:
        return 0 
    else:
        return expSum / ZSum

# 训练
def Train(objSeq,eta,e,times):
    # generate W randomly
    W = np.random.rand(len(objSeq)+1)
    for i in range(times):
        for seq in trainingData:
            dOW = getFai(objSeq,seq) - getSum(objSeq[-1],W,objSeq)
            if abs(dOW).all() < e:
                return W
            W = W + eta*(dOW)
    return W

if __name__ == '__main__':
    maxWi,maxPi,maxSeq = np.ones((len(objSeqList[0])+1)),0,0
    # 遍历所有目标序列，计算联合概率
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    for objSeq in objSeqList:
        #  Inference
        Wi = Train(objSeq,0.01,0.005,1000)
        Pi = Wi@getFai(objSeq,objSeq)
        print("objVec: {},    Wi: {},     P(x,y): {} ".format(objSeq,Wi,Pi))
        if maxPi < Pi:
            maxWi,maxPi,maxSeq = Wi,Pi,objSeq

    print("The sequence with the highest probability is \n{}, Wi: {}, P(x,y): {} ".format(maxSeq,maxWi,maxPi))
