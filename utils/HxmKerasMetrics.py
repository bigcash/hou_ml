# -*- coding: utf-8 -*-
"""

@author: lingbao
"""
import pandas as pd
from sklearn.metrics import roc_auc_score


class HxmKerasMetrics:
    def __init__(self, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
        self.names = []
        self.auc = []
        self.log_loss = []
        self.precision_1 = []
        self.recall_1 = []
        self.f1_1 = []
        self.precision_2 = []
        self.recall_2 = []
        self.f1_2 = []
        self.precision_3 = []
        self.recall_3 = []
        self.f1_3 = []
        self.precision_4 = []
        self.recall_4 = []
        self.f1_4 = []
        self.precision_5 = []
        self.recall_5 = []
        self.f1_5 = []
        self.mean_true = []
        self.mean_pred = []
        self.mean_pos_pred = []
        self.mean_neg_pred = []
        self.pcopc = []
        self.thresholds = thresholds

    def precision(self, data, threshold=0.0):
        TP = data[(data['y_true']==1) & (data['y_pred']>=threshold)].shape[0]
        all_pred_pos = data[data['y_pred']>=threshold].shape[0]
        if(all_pred_pos==0):
            return 1
        precision_score = TP/float(all_pred_pos)
        return precision_score

    def recall(self, data, threshold=0.0):
        TP = data[(data['y_true']==1) & (data['y_pred']>=threshold)].shape[0]
        all_pos = data[data['y_true']==1].shape[0]
        recall_score = TP/float(all_pos)
        return recall_score

    def f1(self, data, threshold=0.0):
        precision1 = self.precision(data, threshold)
        if(precision1==0):
            return 0
        recall1 = self.recall(data, threshold)
        if(recall1==0):
            return 0
        f1_score = 2 * precision1 * recall1 /float(precision1 + recall1)
        return f1_score

    def append(self, name, y_true, y_pred):
        self.names.append(name)
        from sklearn.metrics import log_loss
        self.auc.append(roc_auc_score(y_true, y_pred))
        self.log_loss.append(log_loss(y_true, y_pred))
        data = pd.DataFrame()
        data['y_true'] = y_true
        data['y_pred'] = y_pred
        self.mean_true.append(data['y_true'].mean())
        self.mean_pred.append(data['y_pred'].mean())
        df_pos = data[data['y_true'] == 1]
        df_neg = data[data['y_true'] == 0]
        self.mean_pos_pred.append(df_pos['y_pred'].mean())
        self.mean_neg_pred.append(df_neg['y_pred'].mean())
        self.pcopc.append(data['y_true'].mean()/data['y_pred'].mean())
        self.precision_1.append(self.precision(data, self.thresholds[0]))
        self.recall_1.append(self.recall(data, self.thresholds[0]))
        self.f1_1.append(self.f1(data, self.thresholds[0]))
        self.precision_2.append(self.precision(data, self.thresholds[1]))
        self.recall_2.append(self.recall(data, self.thresholds[1]))
        self.f1_2.append(self.f1(data, self.thresholds[1]))
        self.precision_3.append(self.precision(data, self.thresholds[2]))
        self.recall_3.append(self.recall(data, self.thresholds[2]))
        self.f1_3.append(self.f1(data, self.thresholds[2]))
        self.precision_4.append(self.precision(data, self.thresholds[3]))
        self.recall_4.append(self.recall(data, self.thresholds[3]))
        self.f1_4.append(self.f1(data, self.thresholds[3]))
        self.precision_5.append(self.precision(data, self.thresholds[4]))
        self.recall_5.append(self.recall(data, self.thresholds[4]))
        self.f1_5.append(self.f1(data, self.thresholds[4]))

    def to_csv(self, path):
        import csv
        with open(path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(['id','name', 'auc', 'log_loss', 'mean_true', 'mean_pred', 'mean_pos_pred', 'mean_neg_pred', 'pcopc', 'precision_'+str(self.thresholds[0]), 'recall_'+str(self.thresholds[0]), 'f1_'+str(self.thresholds[0]), 'precision_'+str(self.thresholds[1]), 'recall_'+str(self.thresholds[1]), 'f1_'+str(self.thresholds[1]), 'precision_'+str(self.thresholds[2]), 'recall_'+str(self.thresholds[2]), 'f1_'+str(self.thresholds[2]), 'precision_'+str(self.thresholds[3]), 'recall_'+str(self.thresholds[3]), 'f1_'+str(self.thresholds[3]), 'precision_'+str(self.thresholds[4]), 'recall_'+str(self.thresholds[4]), 'f1_'+str(self.thresholds[4])])
            for i in range(len(self.names)):
                writer.writerow([i+1,self.names[i], self.auc[i], self.log_loss[i], self.mean_true[i], self.mean_pred[i], self.mean_pos_pred[i], self.mean_neg_pred[i], self.pcopc[i], self.precision_1[i], self.recall_1[i], self.f1_1[i], self.precision_2[i], self.recall_2[i], self.f1_2[i], self.precision_3[i], self.recall_3[i], self.f1_3[i], self.precision_4[i], self.recall_4[i], self.f1_4[i], self.precision_5[i], self.recall_5[i], self.f1_5[i]])
            f.close()

