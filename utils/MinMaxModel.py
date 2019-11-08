# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:57:31 2018

@author: hxm
"""

import numpy as np
import pickle


class MinMaxModel:
    def __init__(self):
        self.features = []
        self.min_percents = 0
        self.max_percents = 1
        self.min_values = []
        self.max_values = []

    def fit(self, df_data, features=[], min_percents=0, max_percents=1):
        self.features = features
        self.min_percents = min_percents
        self.max_percents = max_percents
        for idx,f in enumerate(self.features):
            self.min_values.append(df_data[f].quantile(self.min_percents))
            self.max_values.append(df_data[f].quantile(self.max_percents))

    def calculate(self, x, max_v, min_v):
        if max_v==min_v:
            return 0
        else:
            a = (x-min_v)/(max_v-min_v)
            if a>1:
                return 1
            else:
                return a

    def transform(self, df_data):
        df = df_data.copy()
        for idx, f in enumerate(self.features):
            df[f] = df[f].apply(lambda x: self.calculate(x,self.max_values[idx],self.min_values[idx]))
        return df
    
    def fit_transform(self, df_data, features=[], min_percents=0, max_percents=1):
        self.fit(df_data, features, min_percents, max_percents)
        self.printDebugString()
        return self.transform(df_data)
    
    def printDebugString(self):
        print("min max features: ")
        print(self.features)

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def load(self, path):
        model = pickle.load(open(path, 'rb'))
        return model

