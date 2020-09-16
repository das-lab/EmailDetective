# -*- coding: utf-8 -*-
import jieba
import re


def getCcNum(data):
    cc = data.get("Cc")
    if not cc:
        return [0, 0]
    else:
        cc_list = cc.split(",")
        numSum = len(cc_list)
        enronNum = 0
        for i in cc_list:
            if "enron" in i:
                enronNum += 1
        return [numSum, numSum - enronNum]


def getToNum(data) :
    to = data.get("To")
    if not to:
        return [0, 0]
    else:
        to_list = to.split(",")
        numSum = len(to_list)
        toNum = 0
        for i in to_list:
            if "enron" in i:
                toNum += 1
        return [numSum, numSum - toNum]


def getSubNum(data):
    sub = data.get("Subject")
    biaodian = re.findall("\W", sub)
    while " " in biaodian:
        biaodian.remove(" ")
    biaodianNum = len(biaodian)
    wordList = list(jieba.cut(sub, cut_all=False))
    wordNum = len(wordList)
    charNum = 0
    upCharNum = 0
    numNum = 0
    for char in sub:
        if char != " ":
            charNum += 1
        if char.isupper():
            upCharNum += 1
        if char.isnumeric():
            numNum += 1
    return [wordNum, charNum, upCharNum, numNum, biaodianNum]

