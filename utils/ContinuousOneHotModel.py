# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:53:46 2018

@author: hxm
连续值特征的按百分比的数值分段后的onehot特征工程，注意输入字段的数据需为连续值，默认分10段。
"""

import pandas as pd
import pickle

class ContinuousOneHotModel:
    def __init__(self):
        self.section_num_dict = {}
        self.section_value_dict = {}
    
    def fit(self, modelName, df_data, colNameList, section_num=10):
        df = df_data.copy()
        for colName in colNameList:
            keyName = modelName + '_' + colName
            self.section_num_dict[keyName] = section_num
            count = df[colName].nunique()
            if(count>1):
                values = []
                for i in range(0,self.section_num_dict[keyName]-1):
                    end = 1.0/self.section_num_dict[keyName]*(i+1)
                    values.append(df[colName].quantile(end))
                valuesSet = set(values)
                if(len(valuesSet)==0):
                    self.section_value_dict[keyName] = []
                else:
                    valueList = list(valuesSet)
                    valueList.sort(reverse = False)
                    self.section_value_dict[keyName] = valueList
            else:
                self.section_value_dict[keyName] = []

    def transform(self, modelName, df_data, colNameList):
        df = df_data.copy()
        for colName in colNameList:
            keyName = modelName + '_' + colName
            values = self.section_value_dict[keyName]
            if(len(values)==0):
                pass
            else:
                for i in range(0, len(values)):
                    start = 0 if(i==0) else values[i-1]
                    end = values[i]
                    colNameNew = colName+'_hxm_'+str(i)
                    df[colNameNew] = df[colName].apply(lambda x: 1 if((x>=start) & (x<end)) else 0)
                colNameNew = colName+'_hxm_'+str(len(values)+1)
                df[colNameNew] = df[colName].apply(lambda x: 1 if(x>=end) else 0)
        return df
        
    def fit_transform(self, modelName, df_data, colNameList, section_num=10):
        self.fit(modelName, df_data, colNameList, section_num)
        self.printDebugString()
        return self.transform(modelName, df_data, colNameList)
    
    def printDebugString(self):
        print(self.section_value_dict)
        
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def load(path):
        model = pickle.load(open(path, 'rb'))
        return model