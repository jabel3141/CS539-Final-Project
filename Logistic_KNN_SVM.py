import pandas as pd
import pickle
from more_itertools import sliced
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def events_add(event_code):
    event_list = ''
    for i in event_code:
        event_list += i + ' '
    event_list = event_list.rstrip()
    return event_list


pickle_in = open("groups.pickle", "rb")
data = pickle.load(pickle_in)


data = data.drop(["title_y"], axis=1)
data = data.dropna()
data.loc[data["game_session"] == '3c96c025b64f4bd9', "group"] = 17690
data = data[data['group'].apply(lambda x: x > 0)]

eventData = data[["group", "event_code"]]
eventData["event_code"] = eventData["event_code"].apply(str)
sequenceData = eventData.groupby(["group"]).sum().reset_index()
sequenceData['event_code'] = sequenceData['event_code'].apply(lambda x: list(sliced(x, 4)))
sequenceData = sequenceData[sequenceData['event_code'].apply(lambda x: len(x) > 10)]
sequenceData['event_code'] = sequenceData['event_code'].apply(events_add)

accuracyData = data[["group", "accuracy_group"]]
accuracyData = accuracyData.groupby(["group"])["accuracy_group"].mean().reset_index()

mergedData =  sequenceData.merge(accuracyData, on=['group'])[['event_code','accuracy_group']]



MAX_NB_WORDS = 200
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(mergedData['event_code'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(mergedData['event_code'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


Y = mergedData['accuracy_group']
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 1)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)
Y_train = pd.DataFrame(data=Y_train)
Y_test = pd.DataFrame(data=Y_test)


logistic = LogisticRegression(solver='liblinear', multi_class='auto').fit(X_train, Y_train)
print(logistic.predict(X_test))
logistic.score(X_test, Y_test)

logistic = LogisticRegression(solver='lbfgs', multi_class='auto').fit(X_train, Y_train)
print(logistic.predict(X_test))
logistic.score(X_test, Y_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
print(knn.predict(X_test))
print(knn.score(X_test, Y_test))

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, Y_train)
print(knn.predict(X_test))
print(knn.score(X_test, Y_test))

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, Y_train)
print(knn.predict(X_test))
print(knn.score(X_test, Y_test))

knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, Y_train)
print(knn.predict(X_test))
print(knn.score(X_test, Y_test))


svm = SVC(kernel='rbf', decision_function_shape='ovr', gamma='auto')
svm.fit(X_train, Y_train)
print(svm.predict(X_test))
svm.score(X_test, Y_test)

svm = SVC(kernel='poly', decision_function_shape='ovr', degree=10, gamma='scale')
svm.fit(X_train, Y_train)
print(svm.predict(X_test))
print(svm.score(X_test, Y_test))


svm = SVC(kernel='sigmoid', decision_function_shape='ovr', gamma='scale')
svm.fit(X_train, Y_train)
print(svm.predict(X_test))
print(svm.score(X_test, Y_test))
