# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
import time
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler
from sklearn.externals import joblib
import os

def timep(line):
    """
    计时器
    """
    print(line + "-----" + time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())))


class bin_zd:
    def __init__(self,features=None,minnum=None,total_bin_path ="",sparse_switch = 1,is_new = 0):
        """
        :param features: 输入的稀疏矩阵特征
        :param minnum: 单个分箱的最小数量（如希望分为500个bin，则可设置此值为int(num/500+1)）
        :param total_bin_path: 分箱列表文件路径，如有则读取，没有则生成后保存至该目录
        :param sparse_switch: 是否稀疏矩阵
        :param is_new:
        """
        self.features = features
        self.minnum = minnum
        self.sparse_switch = sparse_switch
        self.total_bin_path = total_bin_path
        if(total_bin_path != ""):
            if(os.path.exists(total_bin_path) and is_new == 0):
                self.load_bin_list()
            else:
                self.get_bin_list()
                self.save_bin_list()
        else:
            self.get_bin_list()
    def get_bin_list(self):
        self.total_bin_list = []
        for i in range(self.features.shape[1]):
            if(i%100 == 0):
                timep("finished getting num %d bins"%i)
            feature = self.features[:,i].data
            bin_list = self.get_bin_onefeature(-np.sort(-feature), self.minnum, start_list=[])
            self.total_bin_list += [bin_list]
    def save_bin_list(self):
        file = open(self.total_bin_path, "w")
        file.write("\n".join(map(lambda x:" ".join(map(lambda y:str(y),x)),self.total_bin_list)))
        file.close()
    def bin_map(self,x_str_map):
        if(x_str_map==['']):
            return []
        return list(map(lambda y:float(y),x_str_map))
    def load_bin_list(self):
        file = open(self.total_bin_path,"r")
        self.total_bin_list =list(map(lambda x:self.bin_map(x.strip("\n").split(" ")),file.readlines()))
        file.close()
    def get_bin_onefeature(self,feature_desc,minnum=-1,start_list=[]):
        """
        :param feature_desc:某列特征的非零值，经过从大到小排序后的结果，shape应是1*X（X为该列特征的非零值数量）
        :param minnum:分箱的最少数量
        :param start_list:用于递归记录的分箱点list，初始为空
        :return:分箱点的list
        递归过程：
        1）说如果第minnum个数的值为0，则feature_desc中最小的一个非0值作为唯一分箱点
        2）否则，将第minnum个数的值作为一个分箱点，增加到start_list里，并将大于等于该分箱点的数变为0，重新排序后进行递归获取下一个分箱点
        """
        while(1):
            if (minnum == -1):
                minnum = self.minnum
            # print(feature_desc)
            if(len(feature_desc) == 0):
                return start_list
            if(len(feature_desc)<=minnum):
                return start_list+[feature_desc[-1]]
            if(feature_desc[minnum-1]==0):
                if(feature_desc[0]==0):
                    start_list.sort()
                    return start_list
                else:
                    for i in range(minnum-1):
                        if(feature_desc[minnum - i-2]>0):
                            start_list += [feature_desc[minnum - i - 2]]
                            break
                    start_list.sort()
                    return start_list
            else:
                start_list += [feature_desc[minnum-1]]
                feature_desc = feature_desc[feature_desc < feature_desc[minnum-1]]
    def get_index_bin(self,value,bin_list,alreadnum = 1):
        """
        二分法查找某个值的分箱索引号
        :param value:待查找的值
        :param bin_list:分箱点list
        :param alreadnum:二分法记录点
        :return:索引号，int格式
        比如分箱点为[1,10,23]，则小于1的value变为0，[1,10)的value变为1，[10,23)的value变为2，大于23的value变为3
        """
        if(value < bin_list[0]):
            return alreadnum
        if(value >= bin_list[-1]):
            return len(bin_list) + alreadnum
        elif(value < bin_list[1]):
            return 1 + alreadnum
        midnum = int(len(bin_list) / 2)
        if(value < bin_list[midnum]):
            return self.get_index_bin(value,bin_list[0:midnum],alreadnum)
        if(value >= bin_list[midnum]):
            return self.get_index_bin(value,bin_list[midnum:],midnum+alreadnum)
    def find_bin_feature(self,features=None,sparse_switch = 1):
        """
        对稀疏矩阵进行分箱
        :param features: 输入的稀疏矩阵特征
        :return:经过分箱后的稀疏矩阵
        :param sparse_switch: 是否稀疏矩阵
        """
        if(features == None):
            features = self.features
        if(sparse_switch == 1):
            fea_data = features.data
            fea_indptr = features.indptr
            fea_indices = features.indices
            final_data_list = []
            for i in range(len(fea_data)):
                final_data_list += [self.get_index_bin(fea_data[i],self.total_bin_list[fea_indices[i]])]
            return sparse.csr_matrix((np.array(final_data_list),fea_indices,fea_indptr),shape = features.shape)
        else:
            return self.find_bin_feature(sparse.csr_matrix(features))


if __name__ == '__main__':
    a = np.array([[1, 2, 0, 0, 5], [0, 0, 1, 0, 1], [1, 0, 4, 5, 0], [2, 3, 2, 3, 3]])
    b = sparse.csr_matrix(a)
    print(a)
    base = "D:\\CMCC\\test\\"
    testbin = bin_zd(total_bin_path=base + "bin.txt")
    print(testbin.total_bin_list)
    print(testbin.total_bin_list)
    c = testbin.find_bin_feature(b)
    print(c.toarray())


