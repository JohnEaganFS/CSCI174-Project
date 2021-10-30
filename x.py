import numpy as np
import timeit
import time
from multiprocessing import Pool, cpu_count
from numba import jit, njit, prange, cuda
from scipy.linalg import blas as FB

@njit(parallel=True, fastmath=True)
def allInOne(A,B,c):
    m,n = A.shape
    n,p = B.shape
    if m <= c: #base case we dont need to partition
        C = np.full((m, p), 0)
        for i in prange(m):
            for j in range(p):
                for k in range(n):
                    C[i][j] += A[i][k]*B[k][j]
                    
        #C = FB.dgemm(alpha=1., a=A, b=B, trans_b=True)
        #C = np.dot(A,B)
        return C
    if m <= n:
        #axis 0 = rows, axis 1 = columns
        [A1,A2] = np.array_split(A, 2, axis=1)
        [B1,B2] = np.array_split(B, 2, axis=0)
        C = allInOne(A1,B1,c) + allInOne(A2,B2,c)
        return C
    else: #m>n
        [A1,A2] = np.array_split(A, 2, axis=0)
        [B1,B2] = np.array_split(B, 2, axis=1)
        C1 = allInOne(A1,B1,c)
        C2 = allInOne(A1,B2,c)
        C3 = allInOne(A2,B1,c)
        C4 = allInOne(A2,B2,c)
        
        C12 = np.hstack((C1,C2)) #supposed to be append horizontal. not sure which axis to use
        C34 = np.hstack((C3,C4))
        C = np.vstack((C12,C34))
        return C

@njit(parallel=True, fastmath=True)#compile function so it runs as machine code
#@jit(target = "cuda")
def MatrixMultiply(A,B,c):
    m,n = A.shape
    n,p = B.shape
    if m <= c: #base case we dont need to partition
        C = np.full((m, p), 0)
        for i in prange(m):
            for j in range(p):
                for k in range(n):
                    C[i][j] += A[i][k]*B[k][j]
                    
        #C = FB.dgemm(alpha=1., a=A, b=B, trans_b=True)
        #C = np.dot(A,B)
        return C
    return Partition(A,B,c)

@njit(parallel=True, fastmath=True)#compile function to run as machine code and not through interpreter
def Partition(A,B,c):
    m,n = A.shape
    n,p = B.shape # should we be verifying that the A column and the B row are same length instead of assuming
    if m <= n:
        #axis 0 = rows, axis 1 = columns
        [A1,A2] = np.array_split(A, 2, axis=1)
        [B1,B2] = np.array_split(B, 2, axis=0)
        C = MatrixMultiply(A1,B1,c) + MatrixMultiply(A2,B2,c)
        return C
    else: #m>n
        [A1,A2] = np.array_split(A, 2, axis=0)
        [B1,B2] = np.array_split(B, 2, axis=1)
        C1 = MatrixMultiply(A1,B1,c)
        C2 = MatrixMultiply(A1,B2,c)
        C3 = MatrixMultiply(A2,B1,c)
        C4 = MatrixMultiply(A2,B2,c)
        
        C12 = np.hstack((C1,C2)) #supposed to be append horizontal. not sure which axis to use
        C34 = np.hstack((C3,C4))
        C = np.vstack((C12,C34))
        return C

@njit
def numpyMult(A,B):
    return np.dot(A,B)

if __name__ == "__main__":
    row = 3000
    col = 3000
    testNums = [10, 20, 50, 80, 100, 150, 200, 300]
    np.random.seed(42)

    A = np.random.randint(10, size=(row, col))
    B = np.random.randint(10, size=(row, col))
    C = np.full((row, col), 0)
    cores = cpu_count()

    print("Rows:", row)
    print("Cols:", col)
    print("Cores:", cores)
    print("A:\n", A)
    print("B:\n", B)
   
    #print(timeit.timeit("Partition(A,B,row/cores)", globals=globals(), number=1))

    start = time.time()
    result = MatrixMultiply(A,B,row/cores)
    #result = allInOne(A,B,row/cores)
    #result = FB.dgemm(alpha=1., a=A, b=B, trans_b=True)
    end = time.time()
    print("C:\n",result)
    print("Time Taken:", end - start)
    
    start = time.time()
    #temp = 0
    temp = numpyMult(A.astype(float),B.astype(float))
    end = time.time()
    print("C:\n",temp)
    print("Time Taken:", end - start)