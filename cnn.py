
import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold # import KFold

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=3, kernel_size=(1,10), stride=(1,5)) # 1 input channel, 6 output channels, 3 inputs at a time
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=5, kernel_size=(1,10), stride=(1,5)) # 6 input channel, 15 output channels, 3 dimmension of line
        self.fc1 = nn.Linear(3000, 1500) # 15 * num_rows, output num
        self.fc2 = nn.Linear(1500, 1000)
        self.fc3 = nn.Linear(1000, 250)
        self.fc4 = nn.Linear(250, 100)
        self.fc5 = nn.Linear(100, 4)

    def forward(self, x):

        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = x.view(-1, 3000)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def train(net, train):
    re_weight = torch.tensor([3., 100., 10., 2.])
    criterion = nn.CrossEntropyLoss(weight=re_weight)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net = net.float()
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train, 0):
            # get the inputs; data is a list of [inputs, labels]
            labels, inputs = data
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.numpy()
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.unsqueeze(0)
            # forward + backward + optimize
            labels = torch.tensor([labels])
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            print(outputs, labels, loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("finished training")
    return net


def predict(net, test):
    total = 0
    correct = 0
    predictions = []
    with torch.no_grad():
        for data in test:
            labels, inputs = data
            inputs = inputs.numpy()
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.unsqueeze(0)
            outputs = net(inputs.float())

            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
            total += 1
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    return predictions


def make_tensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg


def to_tensor(index, df):
    temp = df[df.installation_id == index]
    temp = temp.drop(['installation_id'], axis=1)
    np_temp = temp.to_numpy()
    blank = np.ones((1, 92)) * -1
    if np_temp.shape[0] < 300:
        #         print(np_temp.shape[0])
        return 1
    for i in range(0, 300 - np_temp.shape[0]):
        np_temp = np.append(np_temp, blank, axis=0)
    tensor = tf.convert_to_tensor(np_temp)

    return tensor


def to_flat(index, df):
    temp = df[df.installation_id == index]
    temp = temp.drop(['installation_id'], axis=1)
    np_temp = temp.to_numpy()

    blank = np.ones((1, 92)) * -1

    for i in range(0, 300 - np_temp.shape[0]):
        np_temp = np.append(np_temp, blank, axis=0)

    np_flat = np_temp.flatten().tolist()

    return np_flat


def normalize_col(col, df):
    mean_col = df[col].dropna().mean()
    max_col = df[col].dropna().max()
    min_col = df[col].dropna().min()

    df[col] = df[col].apply(lambda x: (x - mean_col) / (max_col - min_col))

    return df

def accuracy(prediction, actual):
    correct = 0

    for i in range(len(prediction)):
        if(prediction[i] == actual.iloc[i]):
            correct += 1

    return correct / len(prediction)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
label = pd.read_csv('train_labels.csv')
specs = pd.read_csv('specs.csv')

with_labels = list(label.groupby('installation_id').count().index)
remove_games = list(label.groupby('game_session').count().index)


train = train[train['installation_id'].isin(with_labels)]
train = train[~train['game_session'].isin(remove_games)]

label = label.groupby('installation_id').tail(1).reset_index(drop=True)

merged = train.merge(label, how='left', left_on='installation_id', right_on='installation_id')

train_data = merged[['installation_id','event_code','event_count','game_time','title_x','type','accuracy_group']]

train_data = train_data.groupby('installation_id').tail(300).reset_index(drop=True)

train_data = pd.concat([train_data, pd.get_dummies(train_data['event_code'], prefix='event_code')], axis=1)
train_data = pd.concat([train_data, pd.get_dummies(train_data['title_x'], prefix='title_x')], axis=1)
train_data = pd.concat([train_data, pd.get_dummies(train_data['type'], prefix='type')], axis=1)

# train_data['event_code'] = pd.Categorical(train_data['event_code'])
# train_data['event_code'] = train_data.event_code.cat.codes

# train_data['title_x'] = pd.Categorical(train_data['title_x'])
# train_data['title_x'] = train_data.title_x.cat.codes

# train_data['type'] = pd.Categorical(train_data['type'])
# train_data['type'] = train_data.type.cat.codes


train_data = train_data.set_index('installation_id')

target = train_data[['installation_id', 'accuracy_group']]

target=target.drop_duplicates(subset='installation_id')

train_data = train_data.drop(['accuracy_group'], axis=1)
train_data = train_data.drop(['title_x'], axis=1)
train_data = train_data.drop(['event_code'], axis=1)
# train_data = train_data.drop(['game_time'], axis=1)
train_data = train_data.drop(['type'], axis=1)
# train_data = train_data.drop(['event_count'], axis=1)


# train_data['game_time'] = train_data['game_time']/10000
# train_data = normalize_col('event_code', train_data)
# train_data = normalize_col('event_count', train_data)
# train_data = normalize_col('game_time', train_data)
# train_data = normalize_col('title_x', train_data)
# train_data = normalize_col('type', train_data)

done = []
rows_list = []
# tensor_df = pd.DataFrame(columns=['tensor', 'accuracy_group'])
# print(tensor_df)
pep_rem = 0
tot_people = 0
for index, row in train_data.iterrows():
    id = row['installation_id']
    if id in done:
        pass
    else:
        done.append(id)
        temp_target = target.loc[target['installation_id'] == id]
        temp_target = temp_target.iloc[0]['accuracy_group']
        temp_tensor = to_tensor(id, train_data)

        try:
            if temp_tensor == 1:
                pep_rem += 1
        except:
            temp_dict = {'tensor': temp_tensor, 'accuracy_group': temp_target}
            rows_list.append(temp_dict)

        tot_people += 1

    if index%100000 == 0:
        print(pep_rem)
        print(tot_people)
        print(index)
tensor_df = pd.DataFrame(rows_list)


done = []
rows_list = []
for index, row in train_data.iterrows():
    id = row['installation_id']
    if id in done:
        pass
    else:
        done.append(id)
        temp_target = target.loc[target['installation_id'] == id]
        temp_target = [temp_target.iloc[0]['accuracy_group']]
        temp_flat = to_flat(id, train_data)
        act = temp_target + temp_flat

        temp_dict = {'flat': temp_flat, 'accuracy_group': temp_target}
        rows_list.append(act)

    if index%100000 == 0:
        print(index)
        
flat_df = pd.DataFrame(rows_list)



# In[27]:


net=Net()
kf = KFold(n_splits=5)
np_tensor = tensor_df.to_numpy()
predictions = []

for train_index, test_index in kf.split(np_tensor):
    net=Net()
    x_train = np_tensor[train_index]
    print(type(np_tensor))
    net = train(net, x_train)
    
    temp_predictions = predict(net, np_tensor[test_index])
    print(temp_predictions)
    predictions.append(temp_predictions)
    
print(predictions)

kf = KFold(n_splits=5)
Y = flat_df[0]
X = flat_df.drop([0], axis=1)
predictions = []
k_predictions = []
r_predictions = []

svm = SVC(kernel='rbf', decision_function_shape='ovr', degree=50, gamma='scale')
knn = KNeighborsClassifier(n_neighbors=15)
rfc = RandomForestClassifier(n_estimators=100)

for train_index, test_index in kf.split(flat_df):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]

    print("training")
    rfc.fit(X_train, y_train)
    print("finished_training")
    knn.fit(X_train, y_train)
    print("finished_training")
    svm.fit(X_train, y_train)
    print("finished_training")
    
    pred = knn.predict(X_test) 
    temp_predictions = svm.predict(X_test)
    rfc_pred = rfc.predict(X_test)
#     print(pred)
#     print(temp_predictions)
#     print(rfc_pred)
    acc = accuracy(pred, y_test)
    o_acc = accuracy(temp_predictions, y_test)
    r_acc = accuracy(rfc_pred, y_test)

    print("svm:", acc)
    print("knn:", o_acc)
    print("rf: ", r_acc)
    predictions.append(temp_predictions)
    k_predictions.append(pred)
    r_predictions.append(rfc_pred)

    
print(predictions)
print(k_predictions)
print(r_predictions)


# In[ ]:





# In[ ]:




