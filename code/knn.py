import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import KFold # import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

#PREDICT JUSTICE 1'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeOnePredictions(df_sel, X_sel):
    y_sel = ['justice1Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape
    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)

    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[3])

    return caseIDs, predictions, gts, jc

#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 2'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeTwoPredictions(df_sel, X_sel):

    y_sel = ['justice2Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[4])
    return predictions, gts, jc

#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 3'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeThreePredictions(df_sel, X_sel):
    y_sel = ['justice3Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[5])
    return predictions, gts, jc

#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 4'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeFourPredictions(df_sel, X_sel):
    y_sel = ['justice4Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[6])
    return predictions, gts, jc

#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 5'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeFivePredictions(df_sel, X_sel):
    y_sel = ['justice5Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[7])
    return predictions, gts, jc

#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 6'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeSixPredictions(df_sel, X_sel):
    y_sel = ['justice6Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[8])
    return predictions, gts, jc

#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 7'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeSevenPredictions(df_sel, X_sel):
    y_sel = ['justice7Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[9])

    return predictions, gts, jc

#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 8'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeEightPredictions(df_sel, X_sel):

    y_sel = ['justice8Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[10])

    return predictions, gts, jc
#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT JUSTICE 9'S VOTE
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getJusticeNinePredictions(df_sel, X_sel):
    y_sel = ['justice9Vote']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    #training on 70%, testing on 30%

    print X_train.shape
    print X_test.shape

    rfc = RandomForestClassifier(n_estimators=100, max_depth=None,criterion='entropy')
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    # print Xreg
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    jc = []
    for x in Xreg:
        caseIDs.append(x[0])
        jc.append(x[11])

    return predictions, gts, jc
#------------------------------------------------------------------------------------------------------------------------------------------------------


#PREDICT WHO WILL WIN
#------------------------------------------------------------------------------------------------------------------------------------------------------
def getWinningProbability(df_sel, X_sel):
    y_sel = ['partyWinning']

    X = np.array(df_sel[X_sel])
    y = np.array(df_sel[y_sel])

    print 'Any NaN fields?' + "\n"
    print df_sel[X_sel].isnull().any().any()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 100)
    #training on 70%, testing on 30%

    # kf = KFold(n_splits=5) # Define the split - into 5 folds
    # kf.get_n_splits(X_train) # returns the number of splitting iterations in the cross-validator
    # print(kf)

    print X_train.shape
    print X_test.shape

    knn = KNeighborsClassifier()

    # for train_index, test_index in kf.split(X):
    #     #print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     rfc.fit(X_train,y_train)

        # predictions = rfc.predict(X_test)

    # p, r, f1, s = precision_recall_fscore_support(y_test, y_pred)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    p, r, f1, s = precision_recall_fscore_support(y_test, predictions)


    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(y_test,predictions))

    Xreg = X_test.tolist()
    predictions = predictions.tolist()
    gts = y_test.tolist()
    caseIDs = []
    for x in Xreg:
        caseIDs.append(x[0])
    return predictions, gts
#------------------------------------------------------------------------------------------------------------------------------------------------------


def main():
    df = pd.read_csv('/Users/anaygupta/version1/data/final.csv')
    columns_to_select = ['caseId','issue','issueArea','partyWinning','decisionDirection','splitVote','majVotes','minVotes',
        'justice1Code','justice2Code','justice3Code','justice4Code','justice5Code','justice6Code','justice7Code','justice8Code','justice9Code',
        'justice1Direction','justice2Direction','justice3Direction','justice4Direction','justice5Direction','justice6Direction','justice7Direction','justice8Direction','justice9Direction',
        'justice1Vote','justice2Vote','justice3Vote','justice4Vote','justice5Vote','justice6Vote','justice7Vote','justice8Vote','justice9Vote',
        'justice1Opinion','justice2Opinion','justice3Opinion','justice4Opinion','justice5Opinion','justice6Opinion','justice7Opinion','justice8Opinion','justice9Opinion',
        'justice1Majority','justice2Majority','justice3Majority','justice4Majority','justice5Majority','justice6Majority','justice7Majority','justice8Majority','justice9Majority']
    df_sel = df[columns_to_select]
    print df_sel

    X_sel = ['caseId','issue','issueArea','justice1Code','justice2Code','justice3Code','justice4Code','justice5Code','justice6Code','justice7Code','justice8Code','justice9Code','justice1Direction','justice2Direction','justice3Direction','justice4Direction','justice5Direction','justice6Direction','justice7Direction','justice8Direction','justice9Direction']

    caseIDs, p1, gt1, jc1 = getJusticeOnePredictions(df_sel, X_sel)
    # p2, gt2, jc2 = getJusticeTwoPredictions(df_sel, X_sel)
    # p3, gt3, jc3 = getJusticeThreePredictions(df_sel, X_sel)
    # p4, gt4, jc4 = getJusticeFourPredictions(df_sel, X_sel)
    # p5, gt5, jc5 = getJusticeFivePredictions(df_sel, X_sel)
    # p6, gt6, jc6 = getJusticeSixPredictions(df_sel, X_sel)
    # p7, gt7, jc7 = getJusticeSevenPredictions(df_sel, X_sel)
    # p8, gt8, jc8 = getJusticeEightPredictions(df_sel, X_sel)
    # p9, gt9, jc9 = getJusticeNinePredictions(df_sel, X_sel)
    # print p9
    p, gt = getWinningProbability(df_sel, X_sel)


    output = pd.DataFrame({
    'Case IDs':caseIDs,

    'Code for Justice 1':jc1,
    'Prediction Vote of Justice 1':p1,
    'Actual Vote of Justice 1':gt1,

    'Code for Justice 2':jc2,
    'Prediction Vote of Justice 2':p2,
    'Actual Vote of Justice 2':gt2,

    'Code for Justice 3':jc3,
    'Prediction Vote of Justice 3':p3,
    'Actual Vote of Justice 3':gt3,

    'Code for Justice 4':jc4,
    'Prediction Vote of Justice 4':p4,
    'Actual Vote of Justice 4':gt4,

    'Code for Justice 5':jc5,
    'Prediction Vote of Justice 5':p5,
    'Actual Vote of Justice 5':gt5,

    'Code for Justice 6':jc6,
    'Prediction Vote of Justice 6':p6,
    'Actual Vote of Justice 6':gt6,

    'Code for Justice 7':jc7,
    'Prediction Vote of Justice 7':p7,
    'Actual Vote of Justice 7':gt7,

    'Code for Justice 8':jc8,
    'Prediction Vote of Justice 8':p8,
    'Actual Vote of Justice 8':gt8,

    'Code for Justice 9':jc9,
    'Prediction Vote of Justice 9':p9,
    'Actual Vote of Justice 9':gt9,

    'Prediction for Winning Party':p,
    'Actual Winning Party':gt,
    })
    output.to_csv('/Users/anaygupta/version1/data/output.csv')
    # print p9
    # print gt9

main()
    #
# print output
# print 'Case IDs'
# print X_test['caseId'].head()
#
# print 'Predictions'
# print predictions.head()
#
# print 'Ground Truths'
# print y_test.head()
