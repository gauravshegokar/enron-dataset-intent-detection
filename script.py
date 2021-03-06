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
    
#################
##### TFIDF #####
#################
def getTFIDF(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(data)
    return X
    
###################
### get n grams ###
###################
def getNgrams(txt, n):
    from nltk import ngrams
    ngrams = ngrams(txt.split(), n)
    l = []
    for grams in ngrams:
        l.append('_'.join(map(str,grams)))
    fl = ' '.join(l)
    return fl

## Cleaning train data
trainData, trainLabels = cleanData(trainRaw)


################# Model Building ##################

#####################
#### Naive Bayes ####
#####################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier as NBC

df = pd.DataFrame({"labels": trainLabels, "trainData": trainData})

train, test = train_test_split(df, test_size = 0.25)

training = zip(train["trainData"].tolist() , train["labels"].tolist())
testing = zip(test["trainData"].tolist() , test["labels"].tolist())

## training model
%time model = NBC(training)

%time print(model.accuracy(training))
## getting accuracy of 90%

## Shows important features for detecting intent
print model.show_informative_features()
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

#-----------------------------------------------------------------------#

#####################
####     SVM     ####
#####################
trainData, trainLabels = cleanData(trainRaw)
testData, testLabels = cleanData(testRaw)

### Adding train and test datasets for SVM as SVM requires 
### same dimentions, number of features for training and test set
data = trainData + testData
labels = trainLabels + testLabels

## getting TFIDF matrix
X = getTFIDF(data)
Y = labels

### Spliting data 80% train - 20% test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.2, random_state=42
)

## SVM Model
from sklearn.svm import SVC
## Value of C is calculated using grid search
svm = SVC(C=2300, kernel='rbf')
svm.fit(x_train, y_train)

print "\nSVM"
print "train accuracy", svm.score(x_train, y_train)
## 88.81% accuracy

print "test accuracy", svm.score(x_test, y_test)
## 79.03% accuracy

from sklearn.metrics import confusion_matrix
pred = svm.predict(x_test)
print "Confusion Matrix\n", confusion_matrix(y_test, pred)
# [[466  77]
# [118 269]]


##########################
#### SVM with 2-gram  ####
##########################

## getting 2-grams LM
data2gram = [getNgrams(i,2) for i in data]

## getting TFIDF matrix
X = getTFIDF(data2gram)
Y = labels

### Spliting data 
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.2, random_state=42
)

## SVM Model for 2-gram
from sklearn.svm import SVC
## value of C has been obtained using grid search
svm = SVC(C=11000, kernel='rbf')
svm.fit(x_train, y_train)

print "\nSVM with 2-gram"
print "train accuracy", svm.score(x_train, y_train)
## 95.61% accuracy

print "test accuracy", svm.score(x_test, y_test)
## 78.60% accuracy

from sklearn.metrics import confusion_matrix
pred = svm.predict(x_test)
print "Confusion Matrix\n", confusion_matrix(y_test, pred)
# [[470 128]
#  [ 73 259]]



###########################
##### SVM with 3-gram #####
###########################

## getting 3-grams LM
data3gram = [getNgrams(i,3) for i in data]

## getting TFIDF matrix
X = getTFIDF(data3gram)
Y = labels

### Spliting data 
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.2, random_state=42
)

## SVM Model for 3-gram
from sklearn.svm import SVC
## value of C has been obtained using grid search
svm = SVC(C=23000 ,kernel='rbf')
svm.fit(x_train, y_train)

print "\n3-gram"
print "train accuracy", svm.score(x_train, y_train)
## 99% accuracy

print "test accuracy", svm.score(x_test, y_test) 
## 75.16% accuracy

from sklearn import metrics
pred = svm.predict(x_test)
print "Confusion Matrix\n", metrics.confusion_matrix(y_test, pred)
# [[507  36]
# [195 192]]

