# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:47:33 2018

@author: hxm
类别特征的编码后进行onehot特征工程，注意使用本方法前需要先将dataframe中的空值处理好。
"""
import numpy as np
import pandas as pd
import pickle


class CategoryOneHotModel:
    def __init__(self, col_name_list=[]):
        self.category_code = {}
        self.category_value = {}
        self.new_col_name = []
        self.new_col_index = []
        self.len_sum = 0
        self.colNameList = col_name_list

    def fit(self, df_data):
        df = df_data.copy()
        k = 0
        for colName in self.colNameList:
            values = df[colName].unique()
            valueList = list(values)
            if(valueList.count(np.nan)>0):
                valueList.remove(np.nan)
            if(len(valueList)>0):
                valueList2 = []
                #转化为str后通过set进行去重
                for i in range(0,len(valueList)):
                    valueList2.append(str(valueList[i]))
                valueSet2 = set(valueList2)
                valueList3 = list(valueSet2)
                code = []
                for i in range(0,len(valueList3)):
                    code.append(i)
                self.category_code[colName] = code
                self.category_value[colName] = valueList3
                self.len_sum = self.len_sum + len(valueList3)
                for row_val in valueList3:
                    self.new_col_name.append(colName + '_hxm_' + str(row_val))
                    self.new_col_index.append(k)
                    k = k + 1
    
    def transform(self, df_data):
        df = df_data.copy()
        for colName in self.colNameList:
            values = self.category_value[colName]
            if(values==None):
                pass
            else:
                codes = self.category_code[colName]
                for i in range(0, len(values)):
                    colNameNew = colName+'_hxm_'+str(codes[i])
                    df[colNameNew] = df[colName].apply(lambda x: 1 if(str(x)==values[i]) else 0)
        return df

    def get_row_list(self, x, cate_features):
        row_list = np.zeros([1, self.len_sum])
        for cate in cate_features:
            if (str(x[cate]) != 'nan'):
                tmp = cate + '_hxm_' + str(x[cate])
                target_index = self.new_col_index[self.new_col_name.index(tmp)]
                row_list[0, target_index] = 1
        return row_list

    '''
    性能优化：直接先生成一个全部为0的矩阵，然后根据值的位置去赋值为1，性能应该有提升。
    '''
    def transform2(self, df_data):
        df1 = df_data.apply(lambda x: self.get_row_list(x, self.colNameList), axis=1)
        df2 = pd.DataFrame(np.concatenate(np.array(df1), axis=0), columns=self.new_col_name)
        return df2
    
    def fit_transform(self, df_data):
        self.fit(df_data)
        self.printDebugString()
        return self.transform(df_data)
    
    def printDebugString(self):
        print("category values: ")
        print(self.category_value)

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def load(self, path):
        model = pickle.load(open(path, 'rb'))
        return model
