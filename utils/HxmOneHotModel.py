# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:24:42 2018

@author: hxm

上世纪八十年代，我国相关部门颁布了属于中国自己的年龄划分标准。这个标准，分为五段，即童年、少年、青年、中年、老年。
1 . 童年，0-6周岁。这个年龄段，又区分为：婴儿（0-3个月）；小儿（4个月-2.5岁）；幼儿（2.5岁-6岁）。
2 . 少年，7-17岁。这个年龄段，又区分为：启蒙期（7-10岁）；逆反期（11-14岁）；成长期（15-17岁）。
3 . 青年，18-40岁。这个年龄段，又区分为：青春期（18-28岁）；成熟期（29-40岁）。
4 . 中年，41-65岁。这个年龄段，又区分为：壮实期（41-48岁）；稳健期（49-55岁）；调整期（56-65岁）。
5 . 老年，66岁以后。这个年龄段，又区分为：初老期（66-72岁）；中老期（73-84岁）；年老期（85岁以后）。

"""
import pickle

class HxmOneHotModel:
    def __init__(self):
        self.model_dict = {}

    def transform_age(self, df_data, age_col_name, remove_old_col=True):
        df = df_data.copy()
        df[age_col_name+"_age_1"] = df[age_col_name].apply(lambda x: 1 if(x<=6) else 0)
        df[age_col_name+"_age_2"] = df[age_col_name].apply(lambda x: 1 if((x>=7) & (x<=17)) else 0)
        df[age_col_name+"_age_3"] = df[age_col_name].apply(lambda x: 1 if((x>=18) & (x<=40)) else 0)
        df[age_col_name+"_age_4"] = df[age_col_name].apply(lambda x: 1 if((x>=41) & (x<=65)) else 0)
        df[age_col_name+"_age_5"] = df[age_col_name].apply(lambda x: 1 if(x>=66) else 0)
        if(remove_old_col):
            df.drop([age_col_name], axis=1, inplace=True)
        return df
    
    def transform_age1(self, df_data, age_col_name, remove_old_col=True):
        df = df_data.copy()
        df[age_col_name+"_age_1"] = df[age_col_name].apply(lambda x: 1 if(x<=17) else 0)
        df[age_col_name+"_age_2"] = df[age_col_name].apply(lambda x: 1 if((x>=18) & (x<=28)) else 0)
        df[age_col_name+"_age_3"] = df[age_col_name].apply(lambda x: 1 if((x>=29) & (x<=40)) else 0)
        df[age_col_name+"_age_4"] = df[age_col_name].apply(lambda x: 1 if((x>=41) & (x<=55)) else 0)
        df[age_col_name+"_age_5"] = df[age_col_name].apply(lambda x: 1 if(x>=56) else 0)
        if(remove_old_col):
            df.drop([age_col_name], axis=1, inplace=True)
        return df
        
    def transform_rise_age(self, df_data, age_col_name, remove_old_col=True):
        df = df_data.copy()
        df[age_col_name+"_age_1"] = df[age_col_name].apply(lambda x: 1 if(x<=24) else 0)
        df[age_col_name+"_age_2"] = df[age_col_name].apply(lambda x: 1 if((x>=25) & (x<=29)) else 0)
        df[age_col_name+"_age_3"] = df[age_col_name].apply(lambda x: 1 if((x>=30) & (x<=38)) else 0)
        df[age_col_name+"_age_4"] = df[age_col_name].apply(lambda x: 1 if((x>=39) & (x<=44)) else 0)
        df[age_col_name+"_age_5"] = df[age_col_name].apply(lambda x: 1 if(x>=45) else 0)
        if(remove_old_col):
            df.drop([age_col_name], axis=1, inplace=True)
        return df
        
    def transform_gt0(self, df_data, age_col_name, remove_old_col=False):
        df = df_data.copy()
        df[age_col_name+"_gt0"] = df[age_col_name].apply(lambda x: 1 if(x>0) else 0)
        if(remove_old_col):
            df.drop([age_col_name], axis=1, inplace=True)
        return df
        
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def load(path):
        model = pickle.load(open(path, 'rb'))
        return model