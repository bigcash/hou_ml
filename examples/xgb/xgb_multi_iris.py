from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb
from utils.HxmMetricsPlot import HxmMetrics

iris = load_iris()

df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df1.head())
X = df1.values
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(y_test.shape, y_train.shape)
data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watchlist = [(data_test, 'eval'), (data_train, 'train')]

xgb_params = {
    'seed': 2018,
    'eta': 0.2,
    'colsample_bytree': 0.9,
    'silent': 1,
    'subsample': 0.9,
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 4,
    'gamma':0.1,
    'lambda': 0.1,
    'alpha': 0.1,
    'eval_metric': ['mlogloss'],
    'min_child_weight': 1
}

bst = xgb.train(xgb_params, data_train, num_boost_round=100, evals=watchlist, early_stopping_rounds=20)
y_hat = bst.predict(data_test, output_margin=True)
y_hat2 = bst.predict(data_test)
print(y_hat)
print(y_hat2)
# hm = HxmMetrics(0.5, y_test, y_hat)
# hm.plot_roc()


def my_categorical_accuracy(y_true, y_pred):
    acc = np.equal(y_pred, y_true)
    acc = np.mean(acc)
    return acc

print("accuracy is {}".format(my_categorical_accuracy(y_test, y_hat2)))
