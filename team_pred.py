import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as gbr
import numpy as np

jaysdata = pd.read_csv('data/bluejays.csv', delimiter = ',')

teams = ['redsox', 'orioles', 'rays', 'whitesox', 'indians',
        'tigers', 'astros', 'royals', 'angels', 'twins', 'athletics',
        'mariners', 'rangers', 'dbacks', 'braves', 'cubs', 'reds',
        'rockies', 'dodgers', 'marlins', 'brewers', 'mets', 'phillies',
        'pirates', 'padres', 'giants', 'cardinals', 'nationals']

data = pd.read_csv('data/yankees.csv', delimiter = ',')
traindata = data.drop(['Lg', 'L', 'Year'],axis=1)
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

r_square = model.score(test_xcol, test_ycol)
predictions = model.predict(test_xcol)
print r_square
