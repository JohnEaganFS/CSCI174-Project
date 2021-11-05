import numpy as np
import cupy as cp
import time as time
from scipy.io import mmread
from multiprocessing import Pool, cpu_count
from numba import jit, njit, prange, cuda
from scipy.linalg import blas as FB

def MatrixMultiply(A,B,c):
    m,n = A.shape
    n,p = B.shape
    if m <= c: #base case we dont need to partition
        C = cp.matmul(A,B)
        return C
    return Partition(A,B,c)

def Partition(A,B,c):
    m,n = A.shape
    n,p = B.shape # should we be verifying that the A column and the B row are same length instead of assuming
    if m <= n:
        #axis 0 = rows, axis 1 = columns
        [A1,A2] = cp.array_split(A, 2, axis=1)
        [B1,B2] = cp.array_split(B, 2, axis=0)
        C = MatrixMultiply(A1,B1,c) + MatrixMultiply(A2,B2,c)
        return C
    else: #m>n
        [A1,A2] = cp.array_split(A, 2, axis=0)
        [B1,B2] = cp.array_split(B, 2, axis=1)
        C1 = MatrixMultiply(A1,B1,c)
        C2 = MatrixMultiply(A1,B2,c)
        C3 = MatrixMultiply(A2,B1,c)
        C4 = MatrixMultiply(A2,B2,c)
        
        C12 = cp.hstack((C1,C2)) #supposed to be append horizontal. not sure which axis to use
        C34 = cp.hstack((C3,C4))
        C = cp.vstack((C12,C34))
        return C

def numpyMult(A,B):
    return np.dot(A,B)

def cupyMult(A,B):
    return cp.matmul(A,B)

if __name__ == "__main__":
    row = 10000
    col = 10000
    testNums = [10, 20, 50, 80, 100, 150, 200, 300]
    np.random.seed(42)

    A = cp.random.randint(10, size=(row, col))
    B = cp.random.randint(10, size=(row, col))
    Anp = cp.asnumpy(A)
    Bnp = cp.asnumpy(B)
    #C = np.full((row, col), 0)
    cores = cpu_count()
    
    print("Rows:", row)
    print("Cols:", col)
    print("Cores:", cores)
    print("A:\n", A)
    print("B:\n", B)

    start = time.time()
    result = MatrixMultiply(A,B,row/cores)
    end = time.time()
    print("C:\n",result)
    print("Our Algorithm")
    print("Time Taken:", end - start)

    start = time.time()
    temp = numpyMult(Anp.astype(float),Bnp.astype(float))
    end = time.time()
    #print("C:\n",temp)
    print("Numpy MM")
    print("Time Taken:", end - start)

    start = time.time()
    temp = cupyMult(A,B)
    end = time.time()
    #print("C:\n",temp)
    print("Cupy MM")
    print("Time Taken:", end - start)