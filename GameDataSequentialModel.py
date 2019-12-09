import pandas as pd
import pickle
from sklearn.model_selection import KFold
from more_itertools import sliced
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


def events_add(event_code):
    event_list = ''
    for i in event_code:
        event_list += i + ' '
    event_list = event_list.rstrip()
    return event_list

pickle_in = open("allLabeled.pickle", "rb")
data = pickle.load(pickle_in)
test = pd.read_csv('test.csv')


data = data.drop(["title_y"], axis=1)
data = data.dropna()

eventData = data[["installation_id", "game_session", "event_code"]]
eventData["event_code"] = eventData["event_code"].apply(str)

sequenceData = eventData.groupby(["installation_id", "game_session"]).sum().reset_index()
sequenceData['event_code'] = sequenceData['event_code'].apply(lambda x: list(sliced(x, 4)))
sequenceData = sequenceData[sequenceData['event_code'].apply(lambda x: len(x) > 10)]
sequenceData['event_code'] = sequenceData['event_code'].apply(events_add)


accuracyData = data[["installation_id", "game_session", "accuracy_group"]]
accuracyData = accuracyData.groupby(["installation_id", "game_session"])["accuracy_group"].mean().reset_index()

mergedData =  sequenceData.merge(accuracyData, on=['installation_id','game_session'])[['event_code','accuracy_group']]

testEventData = test[["installation_id", "game_session", "event_code"]]
testEventData["event_code"] = testEventData["event_code"].apply(str)

testSequenceData = testEventData.groupby(["installation_id", "game_session"]).sum().reset_index()
testSequenceData['event_code'] = testSequenceData['event_code'].apply(lambda x: list(sliced(x, 4)))
testSequenceData = testSequenceData[testSequenceData['event_code'].apply(lambda x: len(x) > 10)]
testSequenceData['event_code'] = testSequenceData['event_code'].apply(events_add)
testSequenceData


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
Y = pd.get_dummies(mergedData['accuracy_group']).values
print('Shape of label tensor:', Y.shape)


cv = KFold(n_splits=5, shuffle=False)
for train_index, test_index in cv.split(X):
    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)])

    accuracy = model.evaluate(X_test, Y_test)

    p = model.predict_classes(X_test)
    p = pd.DataFrame(p)

    print('Test set\n  Loss: {:0.5f}\n  Accuracy: {:0.5f}'.format(accuracy[0], accuracy[1]))
    print(p)


# For submitting to Kaggle

# MAX_NB_WORDS = 200
# MAX_SEQUENCE_LENGTH = 400
# EMBEDDING_DIM = 100
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# tokenizer.fit_on_texts(mergedData['event_code'].values)
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
#
# X = tokenizer.texts_to_sequences(mergedData['event_code'].values)
# X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# print('Shape of data tensor:', X.shape)
# Y = pd.get_dummies(mergedData['accuracy_group']).values
# print('Shape of label tensor:', Y.shape)
#
# X2 = tokenizer.texts_to_sequences(testSequenceData['event_code'].values)
# X2 = pad_sequences(X2, maxlen=MAX_SEQUENCE_LENGTH)
# print('Shape of data tensor:', X2.shape)
#
# X_train = X
# Y_train = Y
#
# X_test = X2
#
# print(X_train.shape,Y_train.shape)
# print(X_test.shape)
#
#
# model = Sequential()
# model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
# model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(4, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(X_train, Y_train, epochs= 5, batch_size=64,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)])
#
# result = pd.concat([testSequenceData['installation_id'].reset_index(drop=True), prediction2], axis=1)
# result = result.rename(columns={0: "accuracy_group"})
# result = result.groupby("installation_id").tail(1).reset_index(drop=True)
#
# result.to_csv("result.csv", index=False)