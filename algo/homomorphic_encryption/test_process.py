from algo.homomorphic_encryption.encrypt import PaillierEncrypt
import numpy as np
import time
import multiprocessing


def process_row(row1, pk):
    x = np.array([object()]*100)
    for i in range(100):
        x[i] = pk.encrypt(row1[i])
    return x

if __name__ == '__main__':
    enc = PaillierEncrypt()
    enc.generate_key()
    pk = enc.get_public_key()
    a = np.array([[1] * 100] * 1000)
    e = np.array([object()] * 1000)
    h1 = time.time()
    print(h1)
    pool = multiprocessing.Pool(processes=4)
    for i in range(a.shape[0]):
        row = a[i]
        # print(row)
        e[i] = pool.apply_async(process_row, args=(row, pk))
    pool.close()
    pool.join()
    h2 = time.time()
    # print(e)
    h3 = h2 - h1
    print(h3)
    print(e.shape)
    print(e[0])
    print("#"*100)
    print(e[0].get())
    # for i in range(3):
    #     print(e[0, i].get())
