import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as gbr
import numpy as np
import pickle as pkl
from sklearn.neural_network import MLPRegressor as mlp

jaysdata = pd.read_csv('data/bluejays.csv', delimiter = ',')

teams = ['redsox', 'orioles', 'rays', 'whitesox', 'indians',
        'tigers', 'astros', 'royals', 'angels', 'twins', 'athletics',
        'mariners', 'rangers', 'dbacks', 'braves', 'cubs', 'reds',
        'rockies', 'dodgers', 'marlins', 'brewers', 'mets', 'phillies',
        'pirates', 'padres', 'giants', 'cardinals', 'nationals']

data = pd.read_csv('data/yankees.csv', delimiter = ',')
traindata = data.drop(['Lg', 'L', 'Year'], axis=1)
for team in teams:
    data = pd.read_csv('data/{0}.csv'.format(team), delimiter = ',')
    traindata = traindata.append(data.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)

traindata = traindata.fillna(0)
train_xcol = traindata.drop(['W'], axis=1).drop([0], axis=0).reset_index()

target = traindata['W']
target = target.drop([len(target)-1], axis=0)


loss = 'lad'
learning_rate = 0.05
n_estimators = 500
min_samples_split = 3
max_depth = 10

gbm = gbr(  loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=max_depth)

model = gbm.fit(train_xcol, target)

testdata = jaysdata.drop(['Lg', 'L', 'Year'],axis=1)
testdata = testdata.fillna(0)
test_xcol = testdata.drop(['W'], axis=1).drop([0], axis=0).reset_index()
test_ycol = testdata['W'].drop([len(testdata)-1])

new_r_square = model.score(test_xcol, test_ycol)


hidden_layer_sizes = (100, 10)
mlpnn = mlp(hidden_layer_sizes=hidden_layer_sizes)

mlp_model = mlpnn.fit(train_xcol, target)
mlp_r_square = mlp_model.score(test_xcol, test_ycol)

if mlp_r_square > new_r_square:
    new_r_square = mlp_r_square
    model = mlp_model

with open('model.pkl', 'rb') as handle:
    curmodel = pkl.load(handle)

cur_r_square = curmodel.score(test_xcol, test_ycol)

if new_r_square > cur_r_square:
    with open('model.pkl', 'wb') as handle:
        pkl.dump(model, handle)
    print new_r_square
else :
    print cur_r_square
    print new_r_square
