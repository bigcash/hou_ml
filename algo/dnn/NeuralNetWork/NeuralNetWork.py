# encoding: utf-8
# NeuralNetWork.py https://blog.csdn.net/randy_01/article/details/83313389
import numpy as np;


def logistic(inX):
    return 1 / (1 + np.exp(-inX))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class Neuron:
    '''
    构建神经元单元，每个单元都有如下属性：1.input;2.output;3.back_weight;4.deltas_item;5.weights.
    每个神经元单元更新自己的weights,多个神经元构成layer,形成weights矩阵
    '''

    def __init__(self, len_input):
        # 输入的初始参数，随机取很小的值(<0.1)
        self.weights = np.random.random(len_input) * 0.1
        # 当前实例的输入
        self.input = np.ones(len_input)
        # 对下一层的输出值
        self.output = 1.0
        # 误差项
        self.deltas_item = 0.0
        # 上一次权重增加的量，记录起来方便后面扩展时可考虑增加冲量
        self.last_weight_add = 0

    def calculate_output(self, x):
        # 计算输出值
        self.input = x;
        self.output = logistic(np.dot(self.weights, self.input))
        return self.output

    def get_back_weight(self):
        # 获取反馈差值
        return self.weights * self.deltas_item

    def update_weight(self, target=0, back_weight=0, learning_rate=0.1, layer="OUTPUT"):
        # 更新权重
        if layer == "OUTPUT":
            self.deltas_item = (target - self.output) * logistic_derivative(self.input)
        elif layer == "HIDDEN":
            self.deltas_item = back_weight * logistic_derivative(self.input)

        delta_weight = self.input * self.deltas_item * learning_rate + 0.9 * self.last_weight_add  # 添加冲量
        self.weights += delta_weight
        self.last_weight_add = delta_weight


class NetLayer:
    '''
    网络层封装，管理当前网络层的神经元列表
    '''

    def __init__(self, len_node, in_count):
        '''
        :param len_node: 当前层的神经元数
        :param in_count: 当前层的输入数
        '''
        # 当前层的神经元列表
        self.neurons = [Neuron(in_count) for _ in range(len_node)];
        # 记录下一层的引用，方便递归操作
        self.next_layer = None

    def calculate_output(self, inX):
        output = np.array([node.calculate_output(inX) for node in self.neurons])
        if self.next_layer is not None:
            return self.next_layer.calculate_output(output)
        return output

    def get_back_weight(self):
        return sum([node.get_back_weight() for node in self.neurons])

    def update_weight(self, learning_rate, target):
        layer = "OUTPUT"
        back_weight = np.zeros(len(self.neurons))
        if self.next_layer is not None:
            back_weight = self.next_layer.update_weight(learning_rate, target)
            layer = "HIDDEN"
        for i, node in enumerate(self.neurons):
            target_item = 0 if len(target) <= i else target[i]
            node.update_weight(target=target_item, back_weight=back_weight[i], learning_rate=learning_rate, layer=layer)
        return self.get_back_weight()


class NeuralNetWork:
    def __init__(self, layers):
        self.layers = []
        self.construct_network(layers)
        pass

    def construct_network(self, layers):
        last_layer = None
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            cur_layer = NetLayer(layer, layers[i - 1])
            self.layers.append(cur_layer)
            if last_layer is not None:
                last_layer.next_layer = cur_layer
            last_layer = cur_layer

    def fit(self, x_train, y_train, learning_rate=0.1, epochs=100000, shuffle=False):
        '''''
        训练网络, 默认按顺序来训练
        方法 1：按训练数据顺序来训练
        方法 2: 随机选择测试
        :param x_train: 输入数据
        :param y_train: 输出数据
        :param learning_rate: 学习率
        :param epochs:权重更新次数
        :param shuffle:随机取数据训练
        '''
        indices = np.arange(len(x_train))
        for _ in range(epochs):
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                self.layers[0].calculate_output(x_train[i])
                self.layers[0].update_weight(learning_rate, y_train[i])
        pass

    def predict(self, x):
        return self.layers[0].calculate_output(x)
