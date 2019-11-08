# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:08:27 2018

@author: hxm
当样本的某一列的值，除了np.nan之外，只有一个重复的值了，那么该列对于label没有作用，可直接删除。
"""
import numpy as np
import pandas as pd
import pickle

class FeatureFilterModel:
    def __init__(self):
        self.cols_dict = {}

    def fit(self, df_data, colNameList, modelName='hxm1'):
        df = df_data.copy()
        invalid_cols = []
        for colName in colNameList:
            values = df[colName].unique()
            valueList = list(values)
            if(valueList.count(np.nan)>0):
                valueList.remove(np.nan)
            valueList2 = []
            #转化为str后通过set进行去重
            for i in range(0,len(valueList)):
                valueList2.append(str(valueList[i]))
            valueSet2 = set(valueList2)
            valueList3 = list(valueSet2)
            if(len(valueList3)<=1):
                invalid_cols.append(colName)
        self.cols_dict[modelName] = invalid_cols
    
    def transform(self, df_data, colNameList, modelName='hxm1'):
        df = df_data.copy()
        cols_dict2 = self.cols_dict.copy()
        invalid_cols = cols_dict2.get(modelName)
        if(invalid_cols==None):
            return df
        else:
            for colName in invalid_cols:
                if(colNameList.count(colName)==0):
                    invalid_cols.remove(colName)
            df.drop(invalid_cols, axis=1, inplace=True)
            return df
    
    def fit_transform(self, df_data, colNameList, modelName='hxm1'):
        self.fit(df_data, colNameList, modelName)
        self.printDebugString()
        return self.transform(df_data, colNameList, modelName)
    
    def printDebugString(self):
        print("invalid cols' dict: ")
        print(self.cols_dict)
        
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def load(path):
        model = pickle.load(open(path, 'rb'))
        return model