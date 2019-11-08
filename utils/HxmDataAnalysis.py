# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:25:12 2018

@author: hxm
"""

import pandas as pd
import numpy as np


class HxmDataAnalysis:
    def __init__(self):
        pass

    def analysis(self, df2):
        df = df2.copy()
        shape0 = df.shape[0]
        df_analy = pd.DataFrame()
        cols = list(df.columns)
        cols_notnull = list(df.notnull().sum())
        cols_nunique = df.nunique()
        cols_mode = df.mode(axis=0).iloc[0]
        df_analy['cols'] = cols
        df_analy['notnull'] = list(cols_notnull)
        df_analy['unique'] = list(cols_nunique)
        df_analy['mode'] = list(cols_mode)
        numericCols = df._get_numeric_data().columns
        #categoricalCols = list(set(cols) - set(numericCols))
        cols_min = df[numericCols].min()
        cols_max = df[numericCols].max()
        cols_mean = df[numericCols].mean()
        cols_median = df[numericCols].median()
        cols_std = df[numericCols].std()
        cols_25 = df[numericCols].quantile(.25)
        cols_50 = df[numericCols].quantile(.5)
        cols_75 = df[numericCols].quantile(.75)
        is_int = []
        nonzero = []
        for i in numericCols:
            a = df[i].values
            nonzero.append(np.count_nonzero(a))
            a[np.isnan(a)] = 0
            a[np.isinf(a)] = 0
            b = a.astype(int)
            c = (a-b).sum()
            if c == 0:
                is_int.append(1)
            else:
                is_int.append(0)
        df_num = pd.DataFrame()
        df_num['cols'] = list(numericCols)
        df_num['nonzero'] = list(nonzero)
        df_num['is_int'] = list(is_int)
        df_num['min'] = list(cols_min)
        df_num['quantile25'] = list(cols_25)
        df_num['quantile50'] = list(cols_50)
        df_num['quantile75'] = list(cols_75)
        df_num['max'] = list(cols_max)
        df_num['mean'] = list(cols_mean)
        df_num['median'] = list(cols_median)
        df_num['std'] = list(cols_std)
        df_analy2 = pd.merge(df_analy, df_num, how='left', on='cols')
        return df_analy2
