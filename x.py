import numpy as np
import timeit
from multiprocessing import Pool

def Partition(A,B):
    m,n = A.shape
    n,p = B.shape # should we be verifiying that the A column and the B row are same length instead of assuming
    if m <= n:
        #axis 0 = rows, axis 1 = columns
        A1,A2 = np.split(A, n//2, axis=1)
        B1,B2 = np.split(B, n//2, axis=0)
        C = MatrixMultiply(A1,B1)+ MatrixMultiply(A2,B2)
        return C
    else: #m>n
        A1,A2 = np.split(A, m//2, axis=0)
        B1,B2 = np.split (B, p//2, axis=1)
        C1 = MatrixMultiply(A1,B1)
        C2 = MatrixMultiply(A1,B2)
        C3 = MatrixMultiply(A2,B1)
        C4 = MatrixMultiply(A2,B2)
        C12 = np.append(C1,C2,axis=0) #suposed to be append horizontal. not sure which axis to use
        c34 = np.append(C3,C4,axis=0)
        C = np.append(C12,C34,axis=1)
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
