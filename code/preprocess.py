import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler

file1 = '/Users/anaygupta/version1/data/scotus.csv'
file2 = '/Users/anaygupta/version1/data/scotusJustice.csv'

df = pd.read_csv(file1)
df2 = pd.read_csv(file2)

#add certReason
scotus = df[['caseId','caseName','issue','issueArea','decisionDirection','partyWinning','splitVote','majVotes','minVotes','petitioner','respondent','jurisdiction','threeJudgeFdc','caseOrigin','caseSource','lcDisagreement','lcDisposition','lcDispositionDirection']]
# print df2['justiceName'][0]
w = 8894
h=9
resultNames = []
resultCodes = []
resultDir = []
resultVote = []
resultOpinion = []
resultMajority = []
result = []
matches = []
# print scotus['caseId'].head(2)
match = 1
for k,i in enumerate(scotus['caseId']):
    counter = 0
    strname = ''
    strcode = ''
    strdir = ''
    strvote = ''
    stropinion = ''
    strmajority = ''
    for key, value in enumerate(df2['caseId']):
        # if(counter < 9):
        if(value == i):
            # match = match +1
            # matches.append(match)
            resultNames.append(df2['justiceName'][key])
            resultCodes.append(df2['justice'][key])
            resultDir.append(df2['direction'][key])
            resultVote.append(df2['vote'][key])
            resultOpinion.append(df2['opinion'][key])
            resultMajority.append(df2['majority'][key])
                # resultNames[k][key] = (df2['justiceName'][key])
                # print strname
                # print strcode
                # print strdir
                # print stropinion
                # print strmajority
                # counter = counter+1
    # resultNames.append(strname)
    # resultCodes.append(strcode)
    # resultDir.append(strdir)
    # resultVote.append(strvote)
    # resultOpinion.append(stropinion)
    # resultMajority.append(strmajority)
    # print "\n"
    # for ind, val in enumerate(resultNames):
    #     scotus['justice' + str(ind+1) + 'name'] = val
    #     scotus['justice' + str(ind+1) + 'code'] = resultCodes[ind]
    #     scotus['justice' + str(ind+1) + 'direction'] = resultDir[ind]
    #     scotus['justice' + str(ind+1) + 'vote'] = resultVote[ind]
    #     scotus['justice' + str(ind+1) + 'opinion'] = resultOpinion[ind]
    #     scotus['justice' + str(ind+1) + 'majority'] = resultMajority[ind]
print matches
j1 = []
jc1 =[]
jd1 =[]
jv1 =[]
jo1 =[]
jm1 =[]

j2 = []
jc2 =[]
jd2 =[]
jv2 =[]
jo2 =[]
jm2 =[]

j3 = []
jc3 =[]
jd3 =[]
jv3 =[]
jo3 =[]
jm3 =[]

j4 = []
jc4 =[]
jd4 =[]
jv4 =[]
jo4 =[]
jm4 =[]

j5 = []
jc5 =[]
jd5 =[]
jv5 =[]
jo5 =[]
jm5 =[]

j6 = []
jc6 =[]
jd6 =[]
jv6 =[]
jo6 =[]
jm6 =[]

j7 = []
jc7 =[]
jd7 =[]
jv7 =[]
jo7 =[]
jm7 =[]

j8 = []
jc8 =[]
jd8 =[]
jv8 =[]
jo8 =[]
jm8 =[]

j9 = []
jc9 =[]
jd9 =[]
jv9 =[]
jo9 =[]
jm9 = []

print resultDir
print resultNames
count = 1
current = 9
for key,r in enumerate(resultNames):
    if count%current == 1:
        j1.append(r)
        jc1.append(resultCodes[key])
        jd1.append(resultDir[key])
        jv1.append(resultVote[key])
        jo1.append(resultOpinion[key])
        jm1.append(resultMajority[key])
    elif count%current == 2:
        j2.append(r)
        jc2.append(resultCodes[key])
        jd2.append(resultDir[key])
        jv2.append(resultVote[key])
        jo2.append(resultOpinion[key])
        jm2.append(resultMajority[key])
    elif count%current == 3:
        j3.append(r)
        jc3.append(resultCodes[key])
        jd3.append(resultDir[key])
        jv3.append(resultVote[key])
        jo3.append(resultOpinion[key])
        jm3.append(resultMajority[key])
    elif count%current == 4:
        j4.append(r)
        jc4.append(resultCodes[key])
        jd4.append(resultDir[key])
        jv4.append(resultVote[key])
        jo4.append(resultOpinion[key])
        jm4.append(resultMajority[key])
    elif count%current == 5:
        j5.append(r)
        jc5.append(resultCodes[key])
        jd5.append(resultDir[key])
        jv5.append(resultVote[key])
        jo5.append(resultOpinion[key])
        jm5.append(resultMajority[key])
    elif count%current == 6:
        j6.append(r)
        jc6.append(resultCodes[key])
        jd6.append(resultDir[key])
        jv6.append(resultVote[key])
        jo6.append(resultOpinion[key])
        jm6.append(resultMajority[key])
    elif count%9 == 7:
        j7.append(r)
        jc7.append(resultCodes[key])
        jd7.append(resultDir[key])
        jv7.append(resultVote[key])
        jo7.append(resultOpinion[key])
        jm7.append(resultMajority[key])
    elif count%9 == 8:
        j8.append(r)
        jc8.append(resultCodes[key])
        jd8.append(resultDir[key])
        jv8.append(resultVote[key])
        jo8.append(resultOpinion[key])
        jm8.append(resultMajority[key])
    elif count%9 == 0:
        j9.append(r)
        jc9.append(resultCodes[key])
        jd9.append(resultDir[key])
        jv9.append(resultVote[key])
        jo9.append(resultOpinion[key])
        jm9.append(resultMajority[key])
    count = count+1
# print j1
# print j2
# print j3
# print j4
# print j5
# print j6
# print j7
# print j8
# print j9
scotus['justice1'] = pd.Series(j1)
scotus['justice2'] = pd.Series(j2)
scotus['justice3'] = pd.Series(j3)
scotus['justice4'] = pd.Series(j4)
scotus['justice5'] = pd.Series(j5)
scotus['justice6'] = pd.Series(j6)
scotus['justice7'] = pd.Series(j7)
scotus['justice8'] = pd.Series(j8)
scotus['justice9'] = pd.Series(j9)

scotus['justice1Code'] = pd.Series(jc1)
scotus['justice2Code'] = pd.Series(jc2)
scotus['justice3Code'] = pd.Series(jc3)
scotus['justice4Code'] = pd.Series(jc4)
scotus['justice5Code'] = pd.Series(jc5)
scotus['justice6Code'] = pd.Series(jc6)
scotus['justice7Code'] = pd.Series(jc7)
scotus['justice8Code'] = pd.Series(jc8)
scotus['justice9Code'] = pd.Series(jc9)


scotus['justice1Direction'] = pd.Series(jd1)
scotus['justice2Direction'] = pd.Series(jd2)
scotus['justice3Direction'] = pd.Series(jd3)
scotus['justice4Direction'] = pd.Series(jd4)
scotus['justice5Direction'] = pd.Series(jd5)
scotus['justice6Direction'] = pd.Series(jd6)
scotus['justice7Direction'] = pd.Series(jd7)
scotus['justice8Direction'] = pd.Series(jd8)
scotus['justice9Direction'] = pd.Series(jd9)


scotus['justice1Vote'] = pd.Series(jv1)
scotus['justice2Vote'] = pd.Series(jv2)
scotus['justice3Vote'] = pd.Series(jv3)
scotus['justice4Vote'] = pd.Series(jv4)
scotus['justice5Vote'] = pd.Series(jv5)
scotus['justice6Vote'] = pd.Series(jv6)
scotus['justice7Vote'] = pd.Series(jv7)
scotus['justice8Vote'] = pd.Series(jv8)
scotus['justice9Vote'] = pd.Series(jv9)

scotus['justice1Opinion'] = pd.Series(jo1)
scotus['justice2Opinion'] = pd.Series(jo2)
scotus['justice3Opinion'] = pd.Series(jo3)
scotus['justice4Opinion'] = pd.Series(jo4)
scotus['justice5Opinion'] = pd.Series(jo5)
scotus['justice6Opinion'] = pd.Series(jo6)
scotus['justice7Opinion'] = pd.Series(jo7)
scotus['justice8Opinion'] = pd.Series(jo8)
scotus['justice9Opinion'] = pd.Series(jo9)


scotus['justice1Majority'] = pd.Series(jm1)
scotus['justice2Majority'] = pd.Series(jm2)
scotus['justice3Majority'] = pd.Series(jm3)
scotus['justice4Majority'] = pd.Series(jm4)
scotus['justice5Majority'] = pd.Series(jm5)
scotus['justice6Majority'] = pd.Series(jm6)
scotus['justice7Majority'] = pd.Series(jm7)
scotus['justice8Majority'] = pd.Series(jm8)
scotus['justice9Majority'] = pd.Series(jm9)





# j1 = []
# j2 = []
# j3 = []
# j4 = []
# j5 = []
# j6 = []
# j7 = []
# j8 = []
# j9 = []
#
# for key, value in enumerate(resultNames):
#     temp = value.split('  ')
#     j1.append()
#     j2.append(temp[1])
#     # scotus['justice1'] = temp[0]
#     # scotus['justice2'] = temp[1]
#     # scotus['justice3'] = temp[2]
#     # scotus['justice4'] = temp[3]
#     # scotus['justice5'] = temp[4]
#     # scotus['justice6'] = temp[5]
#     # scotus['justice7'] = temp[6]
#     # scotus['justice8'] = temp[7]
#     # scotus['justice9'] = temp[8]
# # scotus['justice2'] = [j2]
# print j2[0]
# print scotus['justice2']
# print scotus['justice1']



scotus.to_csv('/Users/anaygupta/version1/data/final.csv')
