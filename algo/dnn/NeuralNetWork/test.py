import numpy as np
import NeuralNetWork as nw

if __name__ == '__main__':
    print("test neural network")

    data = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])

    np.set_printoptions(precision=3, suppress=True)


    for i in range(10):
        network = nw.NeuralNetWork([8, 20, 8])
        # 让输入数据与输出数据相等
        network.fit(data, data, learning_rate=0.1, epochs=150)

        print("\n\n", i, "result")
        for item in data:
            print(item, network.predict(item))
