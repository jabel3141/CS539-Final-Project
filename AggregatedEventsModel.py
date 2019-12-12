import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation

def clean_data(df,specs):
    # too many columns, we'll try it without these
    df = df.drop(['timestamp','event_data','title','type'], axis = 1) 
    # add our own specifications
    df = df.merge(specs, how='inner', on=['event_id'])
    # don't need this after merging
    df = df.drop(['event_id'], axis = 1)

    # sorting...
    df = df.groupby(['installation_id','game_session']).apply(lambda x: x.sort_values(['event_count'])).reset_index(drop=True)
    # now change game_time into a diff with the next event time
    df['game_time'] = df['game_time'].fillna(0)
    df.game_time = df.groupby(['installation_id','game_session'])['game_time'].diff(periods=-1)
    df['game_time'] = df['game_time'].fillna(0).abs()
    # let's ignore events that took more than 10 minutes (even that seems long)
    df = df[df.game_time < 600000]
    return df

def consolidate_features(dataframe):
    cols=['game_session','installation_id','world','accuracy_group','total_time',
          'game_action_time','round_action_time','help_time','movie_time',
          'game_mistake_time','round_mistake_time','rules_time','tutor_time',
          'num_quit','num_skips','num_replays','num_ends','num_rounds']

    sums = [('game_action_time', 6, 1),
            ('round_action_time', 6, 2),
            ('help_time', 3, 2),
            ('movie_time', 5, 0),
            ('game_mistake_time', 2, 1),
            ('round_mistake_time', 2, 2),
            ('rules_time', 5, 3),
            ('tutor_time', 5, 4)]

    nums = [('num_quit', 0, 1),
            ('num_skips', 1, 4),
            ('num_replays', 4, 1),
            ('num_ends', 7, 1),
            ('num_rounds', 5, 2)]
   
    features = pd.DataFrame(columns=cols)
    df = dataframe[['game_session','world','installation_id','accuracy_group']]
    df = df.drop_duplicates()
    features['game_session']=df['game_session']
    features['world']=df['world']
    features['installation_id']=df['installation_id']
    features['accuracy_group']=df['accuracy_group']
    features.set_index('game_session', inplace=True)
    counts = dataframe.groupby(dataframe['game_session']).sum()['game_time'].reset_index()
    counts.rename(columns={"game_time":"total_time"}, inplace=True)
    counts.set_index('game_session', inplace=True)
    features.update(counts)

    for s in sums:
        split = dataframe[dataframe.event_action==s[1]]
        split = split[split.event_type==s[2]]
        counts = split.groupby(dataframe['game_session']).sum()['game_time'].reset_index()
        counts.set_index('game_session', inplace=True)
        counts=counts/features['total_time']
        counts.rename(columns={"game_time":s[0]}, inplace=True)
        features.update(counts)

    for n in nums:
        split = dataframe[dataframe.event_action==n[1]]
        split = split[split.event_type==n[2]]
        counts = split.groupby(dataframe['game_session']).count()['game_time'].reset_index()
        counts.rename(columns={"game_time":n[0]}, inplace=True)
        counts.set_index('game_session', inplace=True)
        features.update(counts)
        
    features = features.fillna(0).reset_index()
    return features



# Import training data
train_labels = pd.read_csv("train_labels.csv")
specs = pd.read_csv("specs_cleaned.csv")
train = pd.read_csv("train.csv")
# filter out entries that didn't complete any assessments
train = train[train.installation_id.isin(train_labels['installation_id'])]   
# attach our labels while we process so we don't lose them
labels = train_labels[['game_session','installation_id','accuracy_group']]
    
result = pd.merge(train, labels, how='outer', on=['game_session','installation_id'])
result = result.dropna()
# get rid of a few columns and sort things
result = clean_data(result,specs)

# filter out failed attempts at assessments - that's what we're trying to predict...
result = result[~result.event_code.isin((4100,4110))]



# aggregate features from data
features = consolidate_features(result)

# during testing, we discovered that half the results are in accuracy_group=3. This throws off the training, so we'll even out our training set.
three = features[features.accuracy_group==3.0]
msk = np.random.rand(len(three)) < 0.65
three = three[msk]
df = features.drop(three.index, errors='ignore', axis=0)
df = pd.get_dummies(df, columns=['world'])


# Split into testing and training data
Y = df['accuracy_group']
X = df.drop(['accuracy_group','game_session','installation_id'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 1)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# normalize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# clean and re-specify the prediction data the same way
test = pd.read_csv("../input/data-science-bowl-2019/test.csv")
test = clean_data(test,specs)
test['accuracy_group'] = 0
pred_data = consolidate_features(test)
result = pred_data[['installation_id']]
pred_data = pd.get_dummies(pred_data, columns=['world'])
pred_data = pred_data.drop(['accuracy_group','game_session','installation_id'], axis=1)
pred_data = sc.transform(pred_data)


# try some models
log1 = LogisticRegression(solver='liblinear', multi_class='auto')
log1.fit(X_train, Y_train)
print("Logistic, liblinear")
print(log1.score(X_test, Y_test))
result['accuracy_group'] = log1.predict(pred_data)
result.to_csv('log1_submission.csv')

log2 = LogisticRegression(solver='lbfgs', multi_class='auto')
log2.fit(X_train, Y_train)
print("Logistic, lbfgs")
print(log2.score(X_test, Y_test))
result['accuracy_group'] = log2.predict(pred_data)
result.to_csv('log2_submission.csv')

knn = KNeighborsClassifier(n_neighbors=150)
knn.fit(X_train, Y_train)
print("KNN, n_neighbors=150")
print(knn.score(X_test, Y_test))
result['accuracy_group'] = knn.predict(pred_data)
result.to_csv('knn_submission.csv')

tree = RandomForestClassifier(n_estimators=50)
tree.fit(X_train, Y_train)
print ("Random Forest, 50 trees")
print (tree.score (X_test, Y_test))
result['accuracy_group'] = tree.predict(pred_data)
result.to_csv('tree_submission.csv')


estimators = []
estimators.append(('log1', log1))
estimators.append(('log2', log2))
estimators.append(('tree', tree))
estimators.append(('knn', knn))

vote = VotingClassifier(estimators)
vote.fit(X_train, Y_train)
print("Voting: 2 logistics, KNN, and Forest")
print(ensemble.score(X_test,Y_test))
result['accuracy_group'] = vote.predict(pred_data)
result.to_csv('vote_submission.csv')
      

nn = Sequential()
nn.add(Dense(units=30, input_shape=(17,)))
nn.add(Activation("relu"))
nn.add(Dense(units=30))
nn.add(Activation("relu"))
nn.add(Dense(units=4))
nn.add(Activation("softmax"))
nn.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
nn.fit(X_train, Y_train, epochs=5)
print("NNet, relu, softmax")
print(nn.evaluate(X_test, Y_test))
result['accuracy_group'] = nn.predict(pred_data)
result.to_csv('nn_submission.csv')


