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
import matplotlib.pyplot as plt

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
        
    #打印roc曲线
    def plot_roc(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='auxiliary')
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    #打印精确度、召回率、F1的曲线
    def plot_threshold_f1(self, section=100):
        sec = 1.0/section
        table = []
        for i in range(0, section):
            threshold = sec*i
            precision1 = self.precision(threshold)
            recall1 = self.recall(threshold)
            f11 = self.f1(threshold)
            table.append([threshold,precision1,recall1,f11])
        df = pd.DataFrame(table)
        df.rename(columns={0:'threshold', 1:'precision', 2:'recall', 3:'f1'},inplace=True)
        thresholds = df['threshold']
        precisions = df['precision']
        recalls = df['recall']
        f1s = df['f1']
        plt.scatter(thresholds, precisions, 10, c='r', label='Precision')
        plt.scatter(thresholds, recalls, 10, c='b', label='Recall')
        plt.scatter(thresholds, f1s, 10, c='k', label='F1')
        plt.title("Threshold Section")
        plt.xlabel("threshold")
        plt.ylabel("value")
        plt.legend()
        plt.show()
        
    #打印在不同阈值区间下，正样本所占的比例的散点图。
    def plot_threshold_pos(self, section=100):
        sec = 1.0/section
        table = []
        for i in range(0, section):
            start = i*sec
            end = (i+1)*sec
            df = self.data[(self.data['y_pred']>=start) & (self.data['y_pred']<end)]
            pos = df[df['y_true']==1].shape[0]
            num = df.shape[0]
            if(num==0):
                rate = 0.0
            else:
                rate = pos/float(num)
            table.append([end, rate])
        df = pd.DataFrame(table)
        df.rename(columns={0:'threshold', 1:'posRate'},inplace=True)
        thresholds = df['threshold']
        posRates = df['posRate']
        plt.scatter(thresholds, posRates, 10, c='r', label='posRate')
        plt.title("Threshold posRate Section")
        plt.xlabel("threshold")
        plt.ylabel("posRate")
        #plt.legend()
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.show()
        
    #打印在不同阈值下，测试样本数量、以及其中正样本数量的散点图
    def plot_threshold_num(self, section=100):
        sec = 1.0/section
        table = []
        for i in range(0, section):
            start = i*sec
            df = self.data[(self.data['y_pred']>=start)]
            pos = df[df['y_true']==1].shape[0]
            num = df.shape[0]
            table.append([start, pos, num])
        df = pd.DataFrame(table)
        df.rename(columns={0:'threshold', 1:'posNum', 2:'totalNum'},inplace=True)
        threshold = df['threshold']
        posNums = df['posNum']
        totalNums = df['totalNum']
        plt.scatter(threshold, posNums, 10, c='r', label='posNum')
        plt.scatter(threshold, totalNums, 10, c='b', label='totalNum')
        plt.title("Threshold posNum Section")
        plt.xlabel("threshold")
        plt.ylabel("number")
        plt.legend()
        #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.show()

    #打印在不同阈值下，分位数、测试样本数量占比、以及其中正样本数量占比的散点图
    def plot_quantile(self, section=20):
        sec = 1.0/section
        table = []
        total = float(self.data.shape[0])
        for i in range(0, section):
            start = i*sec
            quantileNum = self.data['y_pred'].quantile(start)
            df = self.data[(self.data['y_pred']>=start)]
            pos = df[df['y_true']==1].shape[0]
            num = df.shape[0]
            table.append([start, quantileNum, pos/total, num/total])
        df = pd.DataFrame(table)
        df.rename(columns={0:'threshold', 1:'quantile', 2:'posRate', 3:'rate'},inplace=True)
        threshold = df['threshold']
        quantileNum = df['quantile']
        posRate = df['posRate']
        rate = df['rate']
        plt.scatter(threshold, quantileNum, 10, c='r', label='quantile')
        plt.scatter(threshold, posRate, 10, c='b', label='posRate')
        plt.scatter(threshold, rate, 10, c='k', label='rate')
        plt.title("Quantile posNum Section")
        plt.xlabel("threshold")
        plt.ylabel("number")
        plt.legend()
        #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.show()