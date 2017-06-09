# enron-dataset-intent-detection

Intent detection on Enron email set. We define "intent" here to correspond primarily to the categories "request" and "propose". In some cases, we also apply the positive label to some sentences from the "commit" category if they contain datetime, which makes them useful. Detecting the presence of intent in email is useful in many applications, e.g., machine mediation between human and email. The dataset contains parsed sentences from the email along with their intent (either 'yes' or 'no')
Its a 2-class classification problem.

## Model Building

### Naive Bayes
Naive Bayes gives 90% accuracy on training set, 70.3% accuracy on sampled-test set, and 67.3% accuracy on untouched test set.

### SVM
SVM gives 89.19% accuracy on training set, **79.03% accuracy on test set**, which is far better than Naive Bayes.

### SVM with 2-gram
SVM with 2-gram gives 95.61% accuracy on training set, 78.60% accuracy on test set. Due to high number of features generated from 2-gram LM. Model overfits training set giving accuracy of **95.61%**

### SVM with 3-gram
SVM with 3-gram gives 99% accuracy on training set, 75.16% accuracy on test set. Due to **very high number of features** generated from 3-gram LM. Model highly overfits training set giving accuracy of **99%**

## You Need to change paths in script.py to text datasets

