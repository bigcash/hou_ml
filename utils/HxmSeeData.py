# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:25:12 2018

@author: hxm
"""

import pandas as pd

class HxmSeeData:
    def __init__(self, df2):
        self.df = df2.copy()
        
    # 查看pandas的数据中的所有列的去重后数据的个数，主要针对分类数据使用
    def nunique(self, col_list=None):
        if col_list==None:
            col_list = list(self.df)
        result = []
        for i in col_list:
            result.append([i, self.df[i].nunique()])
        #print(result)
        df_result = pd.DataFrame(result)
        df_result.rename(columns={0:'col', 1:'nunique'},inplace=True)
        self.df_nunique = df_result
        return df_result
        
    # 查看某一列去重后的所有值，主要针对分类数据使用
    def unique(self, col_name):
        return self.df[col_name].unique()
    
    # 查看pandas数据中去重后数量大于num的列的数据值情况（每列最多显示10条数据），主要针对分类数据使用
    def showUniqueGt(self, num):
        if(num<=10):
            head = num
        else:
            head = 10
        df_result = self.nunique()
        df_result = df_result[df_result['nunique']>=num].sort_index(by=['nunique'],ascending=False)
        col_list = list(df_result['col'])
        for i in col_list:
            print("Column name: "+i)
            print(self.df[i].unique()[0:head])

    # 某一行数据的不为空的字段的个数
    def not_nan(self, x, cols):
        k = 0
        for col in cols:
            if not (str(x[col]) == 'nan' or str(x[col]) == '' or str(x[col]) == ' '):
                k = k+1
        return k

    '''
    矩阵中每行不为空的字段个数的总体情况
    '''
    def avg_not_nan(self):
        cols = list(self.df.columns)
        self.df['row_not_nan'] = self.df.apply(lambda x: self.not_nan(x, cols), axis=1)
        print("data not nan rows:")
        print("sum: %g", self.df['row_not_nan'].sum())
        print("mean: %g", self.df['row_not_nan'].mean())
        print("min: %g", self.df['row_not_nan'].min())
        print("25%: %g", self.df['row_not_nan'].quantile(0.25))
        print("50%: %g", self.df['row_not_nan'].quantile(.5))
        print("75%: %g", self.df['row_not_nan'].quantile(.75))
        print("max: %g", self.df['row_not_nan'].max())

    '''
    返回矩阵中每行不为空的字段个数大于一定阈值的所有行
    '''
    def not_nan_rows(self, gt_num):
        cols = list(self.df.columns)
        if cols.count('row_not_nan') > 0:
            return self.df[self.df['row_not_nan'] >= gt_num].drop(['row_not_nan'], axis=1)
        else:
            self.df['row_not_nan'] = self.df.apply(lambda x: self.not_nan(x, cols), axis=1)
            return self.df[self.df['row_not_nan'] >= gt_num].drop(['row_not_nan'], axis=1)

    def nan_rows(self, gt_num):
        cols = list(self.df.columns)
        if cols.count('row_not_nan') > 0:
            return self.df[self.df['row_not_nan'] < gt_num].drop(['row_not_nan'], axis=1)
        else:
            self.df['row_not_nan'] = self.df.apply(lambda x: self.not_nan(x, cols), axis=1)
            return self.df[self.df['row_not_nan'] < gt_num].drop(['row_not_nan'], axis=1)
