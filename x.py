import numpy as np
import timeit
import time
from multiprocessing import Pool, cpu_count
from numba import njit, prange

@njit(parallel=True)#compile function so it runs as machine code
def MatrixMultiply(A,B,c):
    m,n = A.shape
    n,p = B.shape
    if m <= c: #base case we dont need to partition
        C = np.full((m, p), 0)
        for i in prange(m):
            for j in range(p):
                for k in range(n):
                    C[i][j] += A[i][k]*B[k][j]
        return C
    return Partition(A,B,c)

@njit#compile function to run as machine code and not through interpreter
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

def RF(A, B, C):
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]

def RFParallel(A, B, C, i):
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[i][j] += A[i][k] * B[k][j]
    return C[i]

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

    start = time.time()
    result = MatrixMultiply(A,B,row/cores)
    end = time.time()
    print("C:\n",result)
    print("Time Taken:", end - start)

    # start = time.time()
    # np.matmul(A,B)
    # end = time.time()
    # print("Numpy matmul time:", end-start)

    # start = time.time()
    # RF(A,B,C)
    # end = time.time()
    # print("Verify Result:", np.array_equal(result, C))
    # print("Serial Time Taken:", end - start)

    #print("Serial:", timeit.timeit('RF(A,B,copyC)', globals=globals(), number=1))

    #pool = Pool()
    #rows = len(A)
    #data = [(A, B, C, i) for i in range(rows)]
    #print("Parallel", timeit.timeit('pool.starmap(RFParallel, data)', globals=globals(), number=1))
    #C = pool.starmap(RFParallel, data)
    #C = [list(array) for array in C]
    #print(C)
