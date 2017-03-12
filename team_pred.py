import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as gbr
import numpy as np

jaysdata = pd.read_csv('bluejays.csv', delimiter = ',')

yankeesdata = pd.read_csv('yankees.csv', delimiter = ',')
redsoxdata = pd.read_csv('redsox.csv', delimiter = ',')
oriolesdata = pd.read_csv('orioles.csv', delimiter = ',')
raysdata = pd.read_csv('rays.csv', delimiter = ',')
whitesoxdata = pd.read_csv('whitesox.csv', delimiter = ',')
indiansdata = pd.read_csv('indians.csv', delimiter = ',')
tigersdata = pd.read_csv('tigers.csv', delimiter = ',')
astrosdata = pd.read_csv('astros.csv', delimiter = ',')
royalsdata = pd.read_csv('royals.csv', delimiter = ',')
angelsdata = pd.read_csv('angels.csv', delimiter = ',')
twinsdata = pd.read_csv('twins.csv', delimiter = ',')
athleticsdata = pd.read_csv('athletics.csv', delimiter = ',')
marinersdata = pd.read_csv('mariners.csv', delimiter = ',')
rangersdata = pd.read_csv('rangers.csv', delimiter = ',')

dbacksdata = pd.read_csv('dbacks.csv', delimiter = ',')
bravesdata = pd.read_csv('braves.csv', delimiter = ',')
cubsdata = pd.read_csv('cubs.csv', delimiter = ',')
redsdata = pd.read_csv('reds.csv', delimiter = ',')
rockiesdata = pd.read_csv('rockies.csv', delimiter = ',')
dodgersdata = pd.read_csv('dodgers.csv', delimiter = ',')
marlinsdata = pd.read_csv('marlins.csv', delimiter = ',')
brewersdata = pd.read_csv('brewers.csv', delimiter = ',')
metsdata = pd.read_csv('mets.csv', delimiter = ',')
philliesdata = pd.read_csv('phillies.csv', delimiter = ',')
piratesdata = pd.read_csv('pirates.csv', delimiter = ',')
padresdata = pd.read_csv('padres.csv', delimiter = ',')
giantsdata = pd.read_csv('giants.csv', delimiter = ',')
cardinalsdata = pd.read_csv('cardinals.csv', delimiter = ',')
nationalsdata = pd.read_csv('nationals.csv', delimiter = ',')

traindata = yankeesdata.drop(['Lg', 'L', 'Year'],axis=1)
traindata = traindata.append(redsoxdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(oriolesdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(raysdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(whitesoxdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(indiansdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(tigersdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(astrosdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(angelsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(twinsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(athleticsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(marinersdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(rangersdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(dbacksdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(bravesdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(cubsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(redsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(rockiesdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(dodgersdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(marlinsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(brewersdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(metsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(philliesdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(piratesdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(padresdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(giantsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(cardinalsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)
traindata = traindata.append(nationalsdata.drop(['Lg', 'L', 'Year'],axis=1)).reset_index().drop(['index'], axis=1)


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
