import numpy as np
import cupy as cp
import time as time
import math
from scipy.io import mmread
from multiprocessing import cpu_count
from scipy.linalg import blas as FB
from pynvml import *

def MatrixMultiply(A,B,c):
    m,n = A.shape
    if m <= c: #base case we dont need to partition
        A = cp.array(A)
        B = cp.array(B)
        return cp.matmul(A,B)
    return Partition(A,B,c)

def Partition(A,B,c):
    m,n = A.shape
    n,p = B.shape
    #h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    #if (15 * m**2 < info.free):
    if (((m*n) + (n*p)) * 8 < info.free):
        #maybe time these
        return PartitionGPU(A,B,c)
    else:
        return PartitionCPU(A,B,c)

def PartitionGPU(A,B,c):
    m,n = A.shape
    #print("GPU")
    if (type(A) == np.ndarray):
        A = cp.array(A)
        B = cp.array(B)
    if m <= n:
        #axis 0 = rows, axis 1 = columns
        [A1,A2] = cp.array_split(A, 2, axis=1)
        [B1,B2] = cp.array_split(B, 2, axis=0)
        C1 = cp.array(MatrixMultiply(A1,B1,c))
        del A1
        del B1
        C2 = cp.array(MatrixMultiply(A2,B2,c))
        del A2
        del B2
        return C1 + C2

        #return cp.array(MatrixMultiply(A1,B1,c)) + cp.array(MatrixMultiply(A2,B2,c))
    else: #m>n
        [A1,A2] = cp.array_split(A, 2, axis=0)
        [B1,B2] = cp.array_split(B, 2, axis=1)

        C1 = MatrixMultiply(A1,B1,c)
        C2 = MatrixMultiply(A1,B2,c)
        C12 = cp.hstack((C1,C2))
        del C1
        del C2

        C3 = MatrixMultiply(A2,B1,c)
        C4 = MatrixMultiply(A2,B2,c)
        C34 = cp.hstack((C3,C4))
        del C3
        del C4
        return cp.vstack((C12,C34))

def PartitionCPU(A,B,c):
    m,n = A.shape
    #print("CPU")
    if (type(A) == cp.ndarray):
        A = cp.asnumpy(A)
        B = cp.asnumpy(B)
    if m <= n:
        #axis 0 = rows, axis 1 = columns
        [A1,A2] = np.array_split(A, 2, axis=1)
        [B1,B2] = np.array_split(B, 2, axis=0)
        C = cp.asnumpy(MatrixMultiply(A1,B1,c)) + cp.asnumpy(MatrixMultiply(A2,B2,c))
        return C
    else: #m>n
        [A1,A2] = np.array_split(A, 2, axis=0)
        [B1,B2] = np.array_split(B, 2, axis=1)
        C1 = cp.asnumpy(MatrixMultiply(A1,B1,c))
        C2 = cp.asnumpy(MatrixMultiply(A1,B2,c))
        C3 = cp.asnumpy(MatrixMultiply(A2,B1,c))
        C4 = cp.asnumpy(MatrixMultiply(A2,B2,c))
        
        C12 = np.hstack((C1,C2))
        C34 = np.hstack((C3,C4))
        C = np.vstack((C12,C34))
        return C

def numpyMult(A,B):
    return np.dot(A,B)

def cupyMult(A,B):
    A = cp.array(A)
    B = cp.array(B)
    return cp.matmul(A,B)

if __name__ == "__main__":
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    cputoGPUCounter = 0
    gputoCPUCounter = 0

    #print(info.free)
    partitionLimit = math.sqrt(info.free / (8)) / 2
    #print(partitionLimit)

    row = 10000
    col = 10000
    testNums = [10, 20, 50, 80, 100, 150, 200, 300]
    np.random.seed(42)

    A = np.random.randint(low=1, high=10, size=(row, col))
    B = np.random.randint(low=1, high=10, size=(row, col))
    cores = cpu_count()
    
    print("Rows:", row)
    print("Cols:", col)
    print("Cores:", cores)
    print("A:\n", A)
    print("B:\n", B)

    start = time.time()
    result = MatrixMultiply(A,B,partitionLimit)
    end = time.time()
    print("C:\n",result)
    print("Our Algorithm")
    print("Time Taken:", end - start)

    # start = time.time()
    # temp = cupyMult(A,B)
    # end = time.time()
    # print("C:\n",temp)
    # print("Cupy MM")
    # print("Time Taken:", end - start)

    # start = time.time()
    # result = FB.dgemm(alpha=1., a=A, b=B, trans_b=True)
    # end = time.time()
    # print("C:\n",result)
    # print("Time Taken:", end - start)

    d = {}
    fileNames = ['datasets/494_bus.mtx', 'datasets/bcsstk17/bcsstk17.mtx', 'datasets/ex11/ex11.mtx', 'datasets/gupta3/gupta3.mtx', 'human_gene1/human_gene1.mtx', 'human_gene2/human_gene2.mtx']
    for fileName in fileNames:
        d[fileName] = 0
        d[fileName + 'Dense'] = 0
    for i in range(10):
        print("Iteration", i)
        for fileName in fileNames:
            print()
            print(fileName)
            #fileName = 'datasets/494_bus.mtx'
            mat = mmread(fileName)        #reads the mtx file
            A = mat.todense(None,None)    #changes the matrix type to numpy.matrix
            A = np.asarray(np.float32(A))
            B = np.asarray(np.float32(A))
            row,col = A.shape
            print("Rows:", row)
            print("Cols:", col)
            # print("Cores:", cores)
            # print("A:\n", A)
            # print("B:\n", B)
            # start = time.time()
            # result = MatrixMultiply(A,B,partitionLimit)
            # end = time.time()
            start = time.time()
            result = FB.dgemm(alpha=1., a=A, b=B, trans_b=True)
            end = time.time()
            #print("CPU to GPU:", cputoGPUCounter)
            #print("GPU to CPU:", gputoCPUCounter)
            # print("C:\n",result)
            # print("Time Taken:", end - start)
            d[fileName] += end - start

            # A = np.random.randn(row, col)
            # A = np.asarray(np.float32(A))
            # B = np.asarray(np.float32(A))
            # start = time.time()
            # result = MatrixMultiply(A,B,partitionLimit)
            # end = time.time()
            # d[fileName + 'Dense'] += end - start

        print("Total (10 iterations):", d)
    for fileName in fileNames:
        d[fileName] /= 10.0
        d[fileName + 'Dense'] /= 10.0
    print("Avg (10 iterations):", d)
