#TODO: first do in simple numpy -> Pytorch -> JAX

import numpy as np
import torch as t
# import jax

def kdetect(mat, ksize, pad, stride):
    #add this stuff to documentation... and this stuff can be improved a lot by adding padding feature and other things like that
    #ksize: tuple{DONE!} , allows the  kernel to skip out and keeps adding
    #pad: affects A
    #stride: should affect loop traversal {DONE!}
    A = np.zeros_like(mat)
    for i in range(0, A.shape[0], stride[0]): 
        for j in range(0, A.shape[1], stride[1]):
            A[i:i+ksize[0], j:j+ksize[1]] += 1

    return A


I = [ np.random.rand(np.random.randint(10),np.random.randint(10)) for i in range(10) ]
O = [(kdetect(j, (2,2), 0, (1,1)), j.shape) for j in I ]

for i in O:
    print(i)