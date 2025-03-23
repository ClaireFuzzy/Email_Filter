import numpy as np
import pandas as pd
import re
import random

def textParse(in_str):
    listofTokens = re.split(r'\W+', in_str)
    return [tok.lower() for tok in listofTokens if len(listofTokens)>=2]

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


