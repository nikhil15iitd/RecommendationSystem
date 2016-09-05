import os
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

datadir = 'C:\\Users\\nikhil\\Documents\\hackerrank-challenge-recommendation-dataset'
challenges = pd.read_csv(os.path.join(datadir, 'challenges.csv'))
submissions = pd.read_csv(os.path.join(datadir, 'submissions.csv'))

challengeidencoder = LabelEncoder().fit(challenges.challenge_id)
contestidencoder = LabelEncoder().fit(challenges.contest_id)
domainencoder = LabelEncoder().fit(challenges.domain)
subdomainencoder = LabelEncoder().fit(challenges.subdomain)
hackeridencoder = LabelEncoder().fit(submissions.hacker_id)

challenges['challengenumber'] = challengeidencoder.transform(challenges.challenge_id)
challenges['contest'] = contestidencoder.transform(challenges.contest_id)
challenges['dom'] = domainencoder.transform(challenges.domain)
challenges['subdom'] = subdomainencoder.transform(challenges.subdomain)
submissions['hackernumber'] = hackeridencoder.transform(submissions.hacker_id)
submissions['challengenumber'] = challengeidencoder.transform(submissions.challenge_id)

challenge_hacker_crossmatrix = np.zeros((len(challengeidencoder.classes_), len(hackeridencoder.classes_)))
challenge_hacker_solved = np.zeros((len(challengeidencoder.classes_), len(hackeridencoder.classes_)))
print(challenge_hacker_crossmatrix.shape)

submissions = submissions.sort_values(['hacker_id', 'challenge_id', 'solved'])
# print(str(submissions.head()))
for index, data in submissions.iterrows():
    challenge_hacker_crossmatrix[data['challengenumber']][data['hackernumber']] = 1
    if data['solved'] == 1:
        challenge_hacker_solved[data['challengenumber']][data['hackernumber']] = 1
    elif data['solved'] == 0:
        challenge_hacker_solved[data['challengenumber']][data['hackernumber']] = -1

corr_mat = np.corrcoef(challenge_hacker_crossmatrix)
print(str(corr_mat.shape))

specified_contestDF = challenges[(challenges['contest_id'] == 'c8ff662c97d345d2')].set_index('challengenumber')
# print(str(specified_contestDF))

high_corr_buckets = {}
for i in range(corr_mat.shape[0]):
    high_corr_buckets[i] = {}
    for j in range(corr_mat.shape[1]):
        if i != j and corr_mat[i][j] > 0.6 and j in specified_contestDF.index:
            high_corr_buckets[i][j] = corr_mat[i][j]

unique_hackers = submissions.hacker_id.unique()

results = {}
for hacker in unique_hackers:
    recommendations = []
    recommendations.append(hacker)
    hackerindex = hackeridencoder.transform(hacker)
    challengeindices = (np.nonzero(challenge_hacker_crossmatrix[:, hackerindex]))[0]
    for challengeindex in challengeindices:
        if challenge_hacker_solved[challengeindex][hackerindex] == -1:
            recommendations.append(challengeidencoder.inverse_transform(challengeindex))
        for i in high_corr_buckets[challengeindex].keys():
            if challengeidencoder.inverse_transform(i) in recommendations or challenge_hacker_solved[i][hackerindex] == 1:
                continue
            recommendations.append(challengeidencoder.inverse_transform(i))
    results[hacker] = recommendations

# post process results
for hacker in results:
    # no challenges
    if len(results[hacker]) == 1:
        solved = submissions[(submissions['hacker_id'] == hacker) & (submissions['solved'] == 1)]
        solved = solved['challenge_id'].tolist()
        newdf = challenges[
            (~challenges['challenge_id'].isin(solved)) & (challenges['contest_id'] == 'c8ff662c97d345d2')].sort_values([
            'total_submissions_count', 'solved_submission_count', 'difficulty'], ascending=[False, False, False])
        values = (newdf['challenge_id'].tolist())[:10]
        values.insert(0, hacker)
        results[hacker] = values
    else:
        challenges_list = (results[hacker])[1:]
        if len(challenges_list) >= 10:
            newdf = challenges[
                (challenges['challenge_id'].isin(challenges_list)) & (
                challenges['contest_id'] == 'c8ff662c97d345d2')].sort_values([
                'total_submissions_count', 'solved_submission_count', 'difficulty'], ascending=[False, False, False])
            values = (newdf['challenge_id'].tolist())[:10]
            values.insert(0, hacker)
            results[hacker] = values
        else:
            # try to recommend 10 challenges if correlated challenges are less
            residual = 10 - len(challenges_list)
            solved = submissions[(submissions['hacker_id'] == hacker) & (submissions['solved'] == 1)]
            solved = solved['challenge_id'].tolist()
            newdf = challenges[
                (~challenges['challenge_id'].isin(solved)) & (~challenges['challenge_id'].isin(challenges_list)) & (
                challenges['contest_id'] == 'c8ff662c97d345d2')].sort_values([
                'total_submissions_count', 'solved_submission_count', 'difficulty'], ascending=[False, False, False])
            values = (newdf['challenge_id'].tolist())[:residual]
            values = challenges_list + values
            values.insert(0, hacker)
            results[hacker] = values


# print(str(results))
with open('recommendation.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in results.items():
        writer.writerow(value)