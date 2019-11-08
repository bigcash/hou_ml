# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
import time
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler
from sklearn.externals import joblib
import os
from sklearn import datasets

def timep(line):
    """
    计时器
    """
    print(line + "-----" + time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time())))


class bin_zd_distance:
    def __init__(self,features,bins,total_bin_path ="",sparse_switch = 1,is_new = 0):
        """
        :param features: 输入的稀疏矩阵特征
        :param minnum: 等距分箱数
        :param total_bin_path: 分箱列表文件路径，如有则直接读取，没有则生成后保存至该目录
        :param sparse_switch: 是否稀疏矩阵
        :param is_new:是否强制写入，默认为0，无影响，如果设为1，则即使total_bin_path路径文件存在，也强制重新生成分箱列表并写入文件
        """
        self.features = features
        self.bins = bins
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
            bin_list = self.get_bin_onefeature(feature, self.bins)
            self.total_bin_list += [bin_list]
    def save_bin_list(self):
        file = open(self.total_bin_path, "w")
        file.write("\n".join(map(lambda x:" ".join(map(lambda y:str(y),x)),self.total_bin_list)))
        file.close()
    def load_bin_list(self):
        file = open(self.total_bin_path,"r")
        self.total_bin_list =list(map(lambda x:x.strip("\n").split(" "),file.readlines()))
        file.close()
    def get_bin_onefeature(self,feature,bins=-1):
        """
        :param feature:某列特征的非零值，shape应是1*X（X为该列特征的非零值数量）
        :param bins:分箱数量
        :return:分箱点的list
        """
        if(len(feature)==0):
            return []
        if(feature.max()==0):
            return []
        if(bins == -1):
            bins = self.bins
        return list(np.array(range(1,bins))*(feature.max()/bins))
    def get_index_bin(self,value,bin_list,alreadnum = 1):
        """
        二分法查找某个值的分箱索引号
        :param value:待查找的值
        :param bin_list:分箱点list
        :param alreadnum:二分法记录点
        :return:索引号，int格式
        比如分箱点为[1,10,23]，则小于1的value变为0，[1,10)的value变为1，[10,23)的value变为2，大于23的value变为3
        """
        if(bin_list==[]):
            if(value>0):
                return 1
            else:
                return 0
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

def get_normal(features):
    return Normalizer().fit_transform(features)

def get_stand(features,path):
    if(os.path.exists(path)):
        staScaler = joblib.load(path)
        return staScaler.transform(features)
    else:
        staScaler = StandardScaler().fit(features)
        joblib.dump(staScaler, path)
        return staScaler.transform(features)

def get_minmax(features,path=""):
    if(os.path.exists(path)):
        max_data = np.loadtxt(path)
    else:
        max_data = features.max(axis = 0).toarray()[0]
        if(path != ""):
            np.savetxt(path,max_data)
    fea_data = []
    count = int(len(features.data)/100)
    for i in range(len(features.data)):
        if(i%count)==0:
            print("start %d/%d minmax"%(i,len(features.data)))
        if(max_data[features.indices[i]]==0):
            fea_data += [(features.data[i]>max_data[features.indices[i]])*1]
        else:
            fea_data += [features.data[i]/max_data[features.indices[i]]]
    return sparse.csr_matrix((fea_data,features.indices,features.indptr))


if __name__ == '__main__':
    a=np.array([[0,0,0,0,0],[0,2,0,0,5],[0,0,1,0,1],[0,0,4,5,0],[0,3,2,3,3]])
    d=sparse.csr_matrix(np.array([[1,1,0,0,0],[1,2,0,0,5],[0,0,1,0,1],[1,0,4,5,0],[2,3,2,3,3]]))
    b=sparse.csr_matrix(a)
    print(a)
    base = "D:\\CMCC\\test\\"
    testbin=bin_zd_distance(features=b,bins=3,total_bin_path=base+"bin.txt",is_new = 1)
    print(testbin.total_bin_list)
    c=testbin.find_bin_feature(b)
    print(c.toarray())
    e=testbin.find_bin_feature(d)
    print("--------------new test---------")
    print(d.toarray())
    print(e.toarray())
