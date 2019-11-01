import numpy as np
from scipy import sparse
import time


class DnnClassifier:
    def __init__(self, learning_rate=0.01, batch_size=-1, layer_list=None, shuffle=False, seed=0):
        self.seed = seed
        self.data_size = None
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        if layer_list is None:
            self.layer_list = [1]
        elif layer_list[-1] != 1:
            self.layer_list = layer_list + [1]
        else:
            self.layer_list = layer_list
        self.layer_num = len(self.layer_list)
        self.batch_size = batch_size
        self.batch_num = None
        self.w = {}
        self.b = {}
        self.opt_w = {}
        self.opt_b = {}
        pass

    def init_params(self, shape_x):
        self.data_size = shape_x[0]
        if self.batch_size == -1:
            self.batch_size = self.data_size
        self.batch_num = self.data_size // self.batch_size + 1
        input_num = shape_x[1]
        for i in range(self.layer_num):
            output_num = self.layer_list[i]
            self.w[i + 1] = np.random.RandomState(self.seed).normal(loc=0.0, scale=0.01,
                                                                    size=[input_num, output_num])
            self.opt_w[i + 1] = np.zeros_like(self.w[i + 1])
            self.b[i + 1] = np.array([0] * output_num)
            self.opt_b[i + 1] = np.zeros_like(self.b[i + 1])
            input_num = output_num

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # def sigmoidDerivationx(self, x):
    #     return x * (1 - x)

    def loss(self, x=None, y=None):
        if x is None:
            #             print(self.pred(self.DATA_x[self.DATA_y[:,0]>0.5]).shape)
            #             print("--------------")
            #             print(np.log(1-self.pred(self.DATA_x[self.DATA_y[:,0]<0.5])).shape)
            #             print("---------")
            #             print(np.concatenate((np.log(self.pred(self.DATA_x[self.DATA_y[:,0]>0.5])),np.log(1-self.pred(self.DATA_x[self.DATA_y[:,0]<0.5]))),axis=0))
            return -np.mean(np.concatenate((np.log(self.pred(self.X[self.y[:, 0] > 0.5])),
                                            np.log(1 - self.pred(self.X[self.y[:, 0] < 0.5]))), axis=0))
        else:
            return -np.mean(np.concatenate((np.log(self.pred(x[y > 0.5])), np.log(1 - self.pred(x[y < 0.5])))))

    def fit(self, x, y):
        self.init_params(x.shape)
        y = y.reshape((self.data_size, 1))
        if self.shuffle:
            indices = np.random.permutation(self.data_size)
        else:
            indices = [i for i in range(self.data_size)]
        print(self.w[1].shape, self.w[2].shape, self.w[3].shape)
        for j in range(self.batch_num):
            start = j * self.batch_size
            end = min(start + self.batch_size, self.data_size)
            batch_ind = indices[start:end]
            a = self.forward(x[batch_ind], y[batch_ind])
            self.update_w_b(a, x[batch_ind], y[batch_ind])

    def update_w_b(self, a, batch_x, batch_y):
        dw_ = {}
        db_ = {}
        current_batch_size = batch_x.shape[0]
        # 根据不同的损失函数、激活函数的求导公式，计算w和b的梯度
        # 根据不同的优化函数更新w和b
        # 以下使用了交叉熵的损失函数、sigmoid的激活函数的导数
        da_last = a[self.layer_num] - batch_y
        print("hxm", a[1].shape, a[2].shape, a[3].shape)
        da = da_last
        db_[self.layer_num] = np.sum(da_last)/current_batch_size
        for i in range(self.layer_num, 1, -1):
            if i == 1:
                dz = da_last
            else:
                dz = a[i] * ( 1 - a[i] ) * da
            print("hxm", da.shape, dz.shape, a[i-1].T.shape, current_batch_size)
            dw = np.dot(dz, a[i-1].T) / current_batch_size
            db = np.mean(dz, axis=1, keepdims=True)
            da_tmp = np.dot(self.w[i].T, dz)
            da = da_tmp
            dw_[i+1] = dw
            db_[i] = db
        dw_[1] = np.dot(batch_x.T, da)
        self.optimizer(dw_, db_)
        # self.w[layer_index] -= self.learning_rate * dw
        # self.b[layer_index] -= self.learning_rate * db
        pass

    def forward(self, batch_x, batch_y):
        a = {}
        z = {}
        z[1] = np.dot(batch_x, self.w[1]) + self.b[1]
        a[1] = self.activation(z[1], 1)
        for i in range(2, self.layer_num+1):
            z[i] = np.dot(a[i-1], self.w[i]) + self.b[i]
            a[i] = self.activation(z[i], i)
        return a

    def optimizer(self, dw_, db_):
        self.sgd(dw_, db_)

    def activation(self, z, layer_num):
        return self.sigmoid(z)

    def sgd(self, dw_, db_):
        for i in range(1, self.layer_num + 1):
            self.w[i] = self.w[i] - self.learning_rate * dw_[i]
            self.b[i] = self.b[i] - self.learning_rate * db_[i]

    def pred(self, x):
        a = self.sigmoid(x * self.w[1] + self.b[1])
        for i in range(1, self.layer_num):
            a = self.sigmoid(np.dot(a, self.w[i + 1]) + self.b[i + 1])
        return a

    def AdaGradOptimizer(self):
        indices = np.random.permutation(self.dataset_size)
        for j in range(int((self.dataset_size - 1) / self.batch_size) + 1):
            start = (j * self.batch_size) % self.dataset_size
            end = min(start + self.batch_size, self.dataset_size)
            self.get_grad(self.DATA_x[start:end], self.DATA_y[start:end])
            for i in range(1, self.layer_num + 1):
                self.opt_W[i] = self.opt_W[i] + np.square(self.grad_w[i])
                self.opt_b[i] = self.opt_b[i] + np.square(self.grad_b[i])
                self.DATA_W[i] = self.DATA_W[i] - self.learn_rate * self.grad_w[i] / (np.sqrt(self.opt_W[i]) + 1e-7)
                self.DATA_b[i] = self.DATA_b[i] - self.learn_rate * self.grad_b[i] / (np.sqrt(self.opt_b[i]) + 1e-7)

    def AdadeltaOptimizer(self):
        pass
