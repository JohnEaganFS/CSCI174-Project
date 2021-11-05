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
        A = cp.array(A)
        B = cp.array(B)
        C = cp.matmul(A,B)
        return cp.asnumpy(C)
    return cp.asnumpy(Partition(A,B,c))

def Partition(A,B,c):
    m,n = A.shape
    n,p = B.shape # should we be verifying that the A column and the B row are same length instead of assuming
    A = cp.array(A)
    B = cp.array(B)
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

    A = np.random.randint(10, size=(row, col))
    B = np.random.randint(10, size=(row, col))
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
    temp = numpyMult(A.astype(float),B.astype(float))
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

    print()
    print("Reading Matrix File Test")
    fileName = 'datasets/494_bus.mtx'
    mat = mmread(fileName)        #reads the mtx file
    A = mat.todense(None,None)    #changes the matrix type to numpy.matrix
    A = cp.array(A)
    B = cp.array(A)
    row,col = A.shape
    print("Rows:", row)
    print("Cols:", col)
    print("Cores:", cores)
    print("A:\n", A)
    print("B:\n", B)
    start = time.time()
    result = MatrixMultiply(A,B,row/cores)
    end = time.time()
    print("C:\n",result)
    print("Time Taken:", end - start)