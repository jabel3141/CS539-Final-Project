import numpy as np
import pandas as pd
import os
import random

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
          'action_time','help_time','movie_time',
          'mistake_time','rules_time','tutor_time',
          'num_quit','num_skips','num_replays','num_ends','num_rounds']

    sums = [('action_time', 6, 2),
            ('help_time', 3, 2),
            ('movie_time', 5, 0),
            ('mistake_time', 2, 2),
            ('rules_time', 5, 3),
            ('tutor_time', 5, 4)]

    nums = [('num_quit', 0, 2),
            ('num_skips', 1, 4),
            ('num_replays', 4, 2),
            ('num_ends', 7, 2),
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
train_labels = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")
specs = pd.read_csv("../input/mystuff/specs_simple.csv")
specs[specs['event_type']==1]['event_type']=2

train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")
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



# aggregate features from game sessions
features = consolidate_features(result)

# during testing, we discovered that half the results are in accuracy_group=3. This throws off the training, so we'll even out our training set.
three = features[features.accuracy_group==3.0]
msk = np.random.rand(len(three)) < 0.65
three = three[msk]
df = features.drop(three.index, errors='ignore', axis=0)

#separate out the labels again
labels = df['accuracy_group']
df = df.drop(['game_session','installation_id','accuracy_group'],axis=1)

#split up into training and testing data
msk = np.random.rand(len(df)) < 0.8
train_d = df[msk].values
train_l = labels[msk].values
test_d = df[~msk].values
test_l = labels[~msk].values

training_data = list(zip(train_d, train_l))
testing_data = list(zip(test_d, test_l))

layer1 = train_d.shape[1]

#organize data for net
train_d = data[msk].values
train_l = labels[msk].values
test_d = data[~msk].values
test_l = labels[~msk].values

training_data = list(zip(train_d, train_l))
testing_data = list(zip(test_d, test_l))

layer1 = train_d.shape[1]

# clean and re-specify the test data the same way
test = pd.read_csv("test.csv")
test = clean_data(test,specs)
test_features=consolidate_features(test)
keep=features[['installation_id','world']]
df = test_features.drop(['installation_id','world'],axis=1)

---------------------------------------------------------------------------

class Network(object):
    def __init__(self,trial,layer1,layer2,layer3):
        self.header = f'{trial},{layer1},{layer2},{layer3},'
        sizes = [layer1,layer2,layer3]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)        
        return a
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        n = len(training_data)
        n_test = len(test_data)
        self.header += f'{epochs},{mini_batch_size},{eta},'
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            test = self.evaluate(test_data)/n_test
            print(self.header + f'{j+1},{test}')
            
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
        
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))

def trial(t,layer1,ep,batch,eta):
    net = Network(t,layer1,layer1,4)
    net.SGD(training_data, epochs=ep, mini_batch_size=batch, eta=eta, test_data=testing_data)
    

