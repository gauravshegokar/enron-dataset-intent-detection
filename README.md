# enron-dataset-intent-detection

Intent detection on Enron email set. We define "intent" here to correspond primarily to the categories "request" and "propose". In some cases, we also apply the positive label to some sentences from the "commit" category if they contain datetime, which makes them useful. Detecting the presence of intent in email is useful in many applications, e.g., machine mediation between human and email. The dataset contains parsed sentences from the email along with their intent (either 'yes' or 'no')
Its a 2-class classification problem.

## Model Building

Naive Bayes Classifier is Used.
NB gives 90% accuracy on training set, 70.3% accuracy on sampled-test set, and 67.3% accuracy on untouched test set.

## You Need to change paths in script.py to text datasets 
