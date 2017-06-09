######################
#### Getting Data ####
######################

with open('./Desktop/stride.ai/data/enron_train.txt') as f:
    train = f.readlines()
trainRaw = [x.strip() for x in train]

with open('./Desktop/stride.ai/data/enron_test.txt') as f:
    test = f.readlines()
testRaw = [x.strip() for x in test]

#######################
#### Cleaning Data ####
#######################

def cleanData(train):
    
    labels = [i.split('\t', 1)[0] for i in train]
    trainData = [i.split('\t', 1)[1] for i in train]

    ## stemming
    from stemming.porter2 import stem
    trainData = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in trainData]

    ## replacing http links with $LINK
    import re
    trainData = [re.sub(r"(<?)http:\S+", "$LINK", i) for i in trainData]

    ## replcaing money with $MONEY
    trainData = [re.sub(r"\$\d+", "$MONEY", i) for i in trainData]

    ## replcaing email ids with $EMAILID
    trainData = [re.sub(r'[\w\.-]+@[\w\.-]+', "$EMAILID", i) for i in trainData]

    ## Lowring the words
    trainData = [i.lower() for i in trainData]

    ## removing punctuations
    import regex as regex
    trainData = [regex.sub(r"[^\P{P}$]+", " ", i) for i in trainData]

    ## remove (unnecessary symbols)
    trainData = [re.sub(r"[^0-9A-Za-z/$' ]", " ", i) for i in trainData]
    
    ## replacing Weekdays with $day
    regString = r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
    trainData = [re.sub(regString, "$days", i) for i in trainData]
    
    ## replacing Months => $month
    regString = r'january|jan|february|feb|march|mar|april|june|july|august|aug|september|sept|october|oct|november|nov|december|dec'
    trainData = [re.sub(regString, "$month", i) for i in trainData]
    
    ## after before during => $time
    regString = r'after|before|during'
    trainData = [re.sub(regString, "$time", i) for i in trainData]
    
    ## replace numbers with $number
    trainData = [re.sub(r'\b\d+\b', "$number", i) for i in trainData]
    
    ## me, her, him ,us or them → $me,
    trainData = [re.sub(r'\b(me|her|him|us|them|you)\b', "$me", i) for i in trainData]
    
    ## striping whitespaces
    trainData = [i.strip() for i in trainData]
    
    return trainData, labels

trainData, trainLabels = cleanData(trainRaw)

########################
#### Model Building ####
########################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier as NBC

df = pd.DataFrame({"labels": trainLabels, "trainData": trainData})

## Partitioning train test datasets (25% data into test dataset)
train, test = train_test_split(df, test_size = 0.25)

## Creating list of tuples
training = zip(train["trainData"].tolist() , train["labels"].tolist())

testing = zip(test["trainData"].tolist() , test["labels"].tolist())


%time model = NBC(training)

from sklearn.externals import joblib

## Saving model
# joblib.dump(model, './Desktop/stride.ai/NBmodel.pkl') 
# model = joblib.load('NBmodel.pkl')

%time print(model.accuracy(training))
## getting accuracy of 90%

model.show_informative_features()
#Most Informative Features
#        contains(please) = True              Yes : No     =     10.0 : 1.0
#            contains(ve) = True               No : Yes    =      9.9 : 1.0
#        contains(verifi) = True              Yes : No     =      9.3 : 1.0
#          contains(sale) = True               No : Yes    =      9.3 : 1.0
#        contains(moment) = True              Yes : No     =      8.6 : 1.0
#       contains(compani) = True               No : Yes    =      7.3 : 1.0
#      contains(deliveri) = True               No : Yes    =      6.9 : 1.0
#    contains(unsubscrib) = True               No : Yes    =      6.9 : 1.0
#        contains(experi) = True               No : Yes    =      6.9 : 1.0
#         contains(remov) = True               No : Yes    =      6.9 : 1.0


%time print(model.accuracy(testing))
## getting accuracy of 70.3%

testData, testLabels = cleanData(testRaw)

testing =  zip(testData , testLabels)

%time print(model.accuracy(testing))
## getting accuracy of 67.3%