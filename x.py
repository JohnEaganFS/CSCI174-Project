import numpy as np
import timeit
from multiprocessing import Pool

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
    row = 300
    col = 300
    testNums = [10, 20, 50, 80, 100, 150, 200, 300]

    A = np.random.randint(10, size=(row, col))
    B = np.random.randint(10, size=(row, col))
    C = np.full((row, col), 0)
    copyC = C.copy()
    print("Rows:", row)
    print("Cols:", col)
    print("Serial:", timeit.timeit('RF(A,B,copyC)', globals=globals(), number=1))

    pool = Pool()
    rows = len(A)
    data = [(A, B, C, i) for i in range(rows)]
    print("Parallel", timeit.timeit('pool.starmap(RFParallel, data)', globals=globals(), number=1))
    #C = pool.starmap(RFParallel, data)
    #C = [list(array) for array in C]
    #print(C)