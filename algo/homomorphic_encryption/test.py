from algo.homomorphic_encryption.encrypt import PaillierEncrypt
import numpy as np
import numexpr as ne
from numba import jit

enc = PaillierEncrypt()
enc.generate_key()
en_fuc = lambda x:enc.get_public_key().encrypt(x)

pk = enc.get_public_key()
# pk1 = pk.copy()
# print(pk, pk1)

a = np.array([[1]*100]*1000)
print(a.shape)

e = np.array([[object()]*100]*1000)
import time
print(time.time())
h1 = time.time()

# @jit
def foo():
    for i in range(a.shape[0]):
        for j in range(100):
            # e[i, j] = enc.get_public_key().encrypt(a[i, j])
            e[i, j] = ne.evaluate("en_fuc(a[i, j])")
# tom_sigmoid_vec = np.vectorize(en_fuc)
# ret = tom_sigmoid_vec(a)
# tom_sigmoid_func = np.frompyfunc(en_fuc, 1, 1)
# ret = tom_sigmoid_func(a)
# print(ret[0, 0])
# foo()

# a = enc.get_public_key().encrypt(1221.52)
# print(a)
# print(a.ciphertext())
#
# b = enc.get_privacy_key().decrypt(a)
#
# print(b)
#
# c = enc.get_public_key().encrypt(2.345)
#
# d = a+c
#
# print(d)
#
# e = enc.get_privacy_key().decrypt(d)
#
# print(e)
# print(1221.52+2.345)
