from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(dir(cancer))

import pandas as pd
df1 = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# print(df1.head())

from dnn_classifier import DnnClassifier
dnn = DnnClassifier(layer_list=[8,2])

X = df1.values
y = cancer.target
dnn.fit(X, cancer.target)
dnn.fit(X, cancer.target)
