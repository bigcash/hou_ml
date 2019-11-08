# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:20:46 2018

@author: hxm

样本的label分为Positive和Negative。
预测成功为True，预测失败为False。
TP: True Positive(真正)将正类预测为正类数
FP: False Positive(假正)将负类预测为正类数
TN: True Negative(真负)将负类预测为负类数
FN: False Negative(假负)将正类预测为负类数
#TP = ((self.y_true==1)*(self.y_pred>=threshold)).sum()
#FP = ((self.y_true==0)*(self.y_pred>=threshold)).sum()
#TN = ((self.y_true==0)*(self.y_pred<threshold)).sum()
#FN = ((self.y_true==1)*(self.y_pred<threshold)).sum()
"""

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss

class HxmMetrics:
    def __init__(self, threshold, y_true, y_pred):
        self.threshold = threshold
        self.y_true = y_true
        self.y_pred = y_pred
        self.log_loss_value = log_loss(self.y_true, self.y_pred)
        self.data = pd.DataFrame()
        self.data['y_true'] = y_true
        self.data['y_pred'] = y_pred
        self.len = self.data.shape[0]
        self.mean_true = self.data['y_true'].mean()
        self.mean_pred = self.data['y_pred'].mean()
        df_pos = self.data[self.data['y_true']==1]
        self.mean_pos_in_pred = df_pos['y_pred'].mean()
        df_neg = self.data[self.data['y_true']==0]
        self.mean_neg_in_pred = df_neg['y_pred'].mean()
        self.pcopc = self.mean_true / self.mean_pred
        self.auc = roc_auc_score(y_true, y_pred)
    
    #控制台输出数据总量、auc、pcopc、测试数据实际平均值、预测值的平均值、预测值中的正样本平均值、预测值中的负样本平均值
    #参数中输入阈值后，显示其该阈值下的准确度、精确度、召回率、F1
    def show(self, threshold=-1):
        print("data count: %d" % self.len)
        print("auc: %.4f" % self.auc)
        print("log loss: %.6f" % self.log_loss_value)
        print("pcopc: %.4f" % self.pcopc)
        print("y_true mean: %.4f" % self.mean_true)
        print("y_pred mean: %.4f" % self.mean_pred)
        print("avg pred of positive: %.4f" % self.mean_pos_in_pred)
        print("avg pred of negative: %.4f" % self.mean_neg_in_pred)
        if(threshold>-1):
            print("accuracy: %.4f" % self.accuracy(threshold))
            print("precision: %.4f" % self.precision(threshold))
            print("recall: %.4f" % self.recall(threshold))
            print("f1: %.4f" % self.f1(threshold))
            num = self.data[self.data['y_pred']>=threshold].shape[0]
            print("Pred >= threshold, num: %d, rate: %.4f" % (num, float(num)/self.len))
        
    def accuracy(self, threshold=-1):
        if(threshold==-1):
            threshold = self.threshold
        TP = self.data[(self.data['y_true']==1) & (self.data['y_pred']>=threshold)].shape[0]
        TN = self.data[(self.data['y_true']==0) & (self.data['y_pred']<threshold)].shape[0]
        accuracy_score = (TP+TN)/float(self.len)
        return accuracy_score

    def precision(self, threshold=-1):
        if(threshold==-1):
            threshold = self.threshold
        TP = self.data[(self.data['y_true']==1) & (self.data['y_pred']>=threshold)].shape[0]
        #FP = self.data[(self.data['y_true']==0) & (self.data['y_pred']>=threshold)].shape[0]
        all_pred_pos = self.data[self.data['y_pred']>=threshold].shape[0]
        if(all_pred_pos==0):
            return 1
        precision_score = TP/float(all_pred_pos)
        return precision_score

    def recall(self, threshold=-1):
        if(threshold==-1):
            threshold = self.threshold
        TP = self.data[(self.data['y_true']==1) & (self.data['y_pred']>=threshold)].shape[0]
        #FN = self.data[(self.data['y_true']==1) & (self.data['y_pred']<threshold)].shape[0]
        all_pos = self.data[self.data['y_true']==1].shape[0]
        recall_score = TP/float(all_pos)
        return recall_score

    def f1(self, threshold=-1):
        if(threshold==-1):
            threshold = self.threshold
        precision1 = self.precision(threshold)
        if(precision1==0):
            return 0
        recall1 = self.recall(threshold)
        if(recall1==0):
            return 0
        f1_score = 2 * precision1 * recall1 /float(precision1 + recall1)
        return f1_score
    
    #FPR代表将负例错分为正例的概率
    def FPR(self, threshold=-1):
        if(threshold==-1):
            threshold = self.threshold
        FP = self.data[(self.data['y_true']==0) & (self.data['y_pred']>=threshold)].shape[0]
        #TN = self.data[(self.data['y_true']==0) & (self.data['y_pred']<threshold)].shape[0]
        all_neg = self.data[self.data['y_true']==0].shape[0]
        return FP/float(all_neg)
        