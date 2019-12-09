import pandas as pd
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
label = pd.read_csv('train_labels.csv')

with_labels = list(label.groupby('installation_id').count().index)
train = train[train['installation_id'].isin(with_labels)]
train = pd.merge(train, label, how='left', on=['installation_id', 'game_session']).reset_index(drop=True)

assessmentLabels = list(label.groupby(["installation_id", "game_session"]).count().index)

timeLabels = []

for install, game in assessmentLabels:
    startTime = \
    train.loc[(train["installation_id"] == install) & (train["game_session"] == game)]["timestamp"].reset_index(
        drop=True)[0]
    timeLabels.append((install, game, startTime))

timeLabels.sort(key = lambda x: x[2])

train["group"] = 0

groupNum = 0
for install, game, time in timeLabels:
    accuracyGroup, numCorrect, numIncorrect, accuracy = \
    train.loc[(train["installation_id"] == install) & (train["game_session"] == game)][
        ["accuracy_group", "num_correct", "num_incorrect", "accuracy"]].reset_index(drop=True).iloc[0]

    rows = (train["timestamp"] < time) & (train["installation_id"] == install) & (train["accuracy_group"].isna())

    train.loc[rows, "group"] = groupNum

    train.loc[rows, "accuracy_group"] = accuracyGroup
    train.loc[rows, "num_correct"] = numCorrect
    train.loc[rows, "num_incorrect"] = numIncorrect
    train.loc[rows, "accuracy"] = accuracy

    groupNum += 1

import pickle
name = "groups.pickle"
pickle_out = open(name, "wb")
pickle.dump(train, pickle_out)
pickle_out.close()



for install, game, time in timeLabels:
    accuracyGroup, numCorrect, numIncorrect, accuracy = \
    train.loc[(train["installation_id"] == install) & (train["game_session"] == game)][
        ["accuracy_group", "num_correct", "num_incorrect", "accuracy"]].reset_index(drop=True).iloc[0]

    rows = (train["timestamp"] < time) & (train["installation_id"] == install) & (train["accuracy_group"].isna())

    train.loc[rows, "accuracy_group"] = accuracyGroup
    train.loc[rows, "num_correct"] = numCorrect
    train.loc[rows, "num_incorrect"] = numIncorrect
    train.loc[rows, "accuracy"] = accuracy

train

import pickle
name = "allLabeled.pickle"
pickle_out = open(name, "wb")
pickle.dump(train, pickle_out)
pickle_out.close()
#data.to_csv("test.csv", index=False