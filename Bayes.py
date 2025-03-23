import numpy as np
import pandas as pd
import re
import random

def textParse(in_str):
    listofTokens = re.split(r'\W+', in_str)
    return [tok.lower() for tok in listofTokens if len(listofTokens)>=2]

def createVocabList(doclist):
    vocabSet = set([])
    for document in doclist:
       vocabSet = vocabSet|set(document)
    return list(vocabSet)

def setOfWord2Vec(vocablist, in_set):
    returnVec = [0]*len(vocablist)
    for word in in_set:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec

def trainNB(trainMat, trainClass):
    num_train = len(trainClass)
    num_words = len(trainMat[0])
    p1 = sum(trainClass)/float(num_train)
    p0Num = np.ones((num_words))
    p1Num = np.ones((num_words)) #拉普拉斯平滑
    p0Den = 2 #通常设置成类别个数
    p1Den = 2

    for i in range(num_train):
        if trainClass[i] == 1:
            p1Num += trainMat[i]
            p1Den += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Den += sum(trainMat[i])
    #值太小了，要log变换
    p1Vec = np.log(p1Num/p1Den)
    p0Vec = np.log(p0Num/p0Den)
    return p0Vec, p1Vec, p1

def classifyNB(wordVec, p0Vec, p1Vec, p1_class):
    #Both sides have log, so multiplication of p1_class and p1_Vec becomes addition.
    #Also, P(D|h+) is a series of multiplication (see on notebook), so it becomes addition as well.
    p1 = np.log(p1_class) + sum(wordVec*p1Vec)
    p0 = np.log(1.0-p1_class) + sum(wordVec*p0Vec)
    if(p0>p1):
        return 0
    else:
        return 1

def spam():
    doclist = []
    classlist = []
    for i in range(1,26):
        wordlist = textParse(open('./email/spam/%d.txt'%i, 'r').read())
        doclist.append(wordlist)
        classlist.append(1) #1 represents spam mails

        wordlist = textParse(open('./email/ham/%d.txt' % i, 'r').read())
        doclist.append(wordlist)
        classlist.append(0)  # 0 represents not spam mails

    vocablist = createVocabList(doclist)
    trainSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del trainSet[randIndex]

    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWord2Vec(vocablist, doclist[docIndex]))
        trainClass.append(classlist[docIndex])
    p0Vec, p1Vec, p1 = trainNB(np.array(trainMat), np.array(trainClass))
    errorcnt = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocablist, doclist[docIndex])
        res = classifyNB(np.array(wordVec), p0Vec, p1Vec, p1)
        if res != classlist[docIndex]:
            errorcnt += 1
    print("Error Rate is:",errorcnt/10.0)

if __name__ == '__main__':
    spam()