import queue
import threading
import time

from algo.homomorphic_encryption.encrypt import PaillierEncrypt
import numpy as np
import numexpr as ne
from numba import jit

enc = PaillierEncrypt()
enc.generate_key()
pk = enc.get_public_key()
thread_num = 20
queue_num = 50
exitFlag = 0
a = np.array([[1]*100]*1000)
print(a.shape)

e = np.array([[object()]*100]*1000)
print(time.time())
h1 = time.time()



class MyThread (threading.Thread):
    def __init__(self, pk1, name, q):
        threading.Thread.__init__(self)
        self.pk1 = pk1
        self.name = name
        self.q = q

    def run(self):
        print("开启线程：" + self.name)
        while not exitFlag:
            queueLock.acquire()
            if not self.q.empty():
                ind, row = self.q.get()
                queueLock.release()
                # print(self.name, ind)
                for j in range(100):
                    e[ind, j] = self.pk1.encrypt(row[j])
            else:
                queueLock.release()
        print("退出线程：" + self.name)

print("test1")
queueLock = threading.Lock()
workQueue = queue.Queue(queue_num)

threads = []
for i in range(thread_num):
    thread = MyThread(pk, "name"+str(i), workQueue)
    thread.start()
    threads.append(thread)

for i in range(a.shape[0]):
    queueLock.acquire()
    if workQueue.full():
        queueLock.release()
        time.sleep(0.5)
    else:
        workQueue.put((i, a[i]))
        queueLock.release()
        # print("put ",i)

# 等待队列清空
while not workQueue.empty():
    pass

# 通知线程是时候退出
exitFlag = 1

# 等待所有线程完成
for t in threads:
    t.join()

print("退出主线程")
print(time.time())
