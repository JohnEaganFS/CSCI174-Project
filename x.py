import numpy as np
import timeit
import time
from scipy.io import mmread
from multiprocessing import Pool, cpu_count
from numba import jit, njit, prange, cuda
from scipy.linalg import blas as FB

@njit(parallel=True, fastmath=True)#compile function so it runs as machine code
def MatrixMultiply(A,B,c):
    m,n = A.shape
    n,p = B.shape
    if m <= c: #base case we dont need to partition
        C = np.full((m, p), 0)
        for i in prange(m):
            for j in range(p):
                for k in range(n):
                    C[i][j] += A[i][k]*B[k][j]
        #return np.dot(A,B)
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

if __name__ == "__main__":
    cores = cpu_count()

    d = {} # Dictionary for storing execution time of each dataset
    fileNames = ['datasets/494_bus.mtx', 'datasets/bcsstk17/bcsstk17.mtx', 'datasets/ex11/ex11.mtx', 'datasets/gupta3/gupta3.mtx', 'human_gene1/human_gene1.mtx', 'human_gene2/human_gene2.mtx']

    for fileName in fileNames: # Initialize times to 0
        d[fileName] = 0
        d[fileName + 'Dense'] = 0
        #d[fileName+'Numpy'] = 0
        #d[fileName+'NumpyDense'] = 0
    for i in range(1): # For 10 iterations (totaling time for 10 and then averaging)
        print("Iteration", i)
        for fileName in fileNames: # for each dataset
            print()
            print(fileName)
            mat = mmread(fileName)        # reads the mtx file
            A = mat.todense(None,None)    # changes the matrix type to numpy.matrix
            A = np.asarray(np.float32(A)) # converting matrices to ndarrays with float32 elements (8 bytes)
            B = np.asarray(np.float32(A))
            row,col = A.shape
            print("Rows:", row)
            print("Cols:", col)

            start = time.time()
            result = MatrixMultiply(A,B,row/cores) # Tracking execution time for our algorithm on the sparse dataset
            end = time.time()
            print("Time Taken:", end - start)
            d[fileName] += end - start # Adding to time dictionary

        print("Current Total:", d)
    # Averaging execution time across all 10 runs
    for fileName in fileNames:
        d[fileName] /= 10.0
        d[fileName + 'Dense'] /= 10.0
        #d[fileName+'Numpy'] /= 10.0
        #d[fileName+'NumpyDense'] /= 10.0
    print("Avg (10 iterations):", d)
    
    # """ 
    # Example reading datafile 
    # We can adjust this to read 2 different data files
    # We just have to make sure rows of col A = row B
    # """
    # fileName = 'datasets/494_bus.mtx'
    # mat = mmread(fileName)        #reads the mtx file
    # A = mat.todense(None,None)    #changes the matrix type to numpy.matrix
    # B = A                         #Another copy to multiply by itself. 
    # row,col = A.shape
    # print("Rows:", row)
    # print("Cols:", col)
    # print("Cores:", cores)
    # print("A:\n", A)
    # print("B:\n", B)
    # start = time.time()
    # result = MatrixMultiply(A,B,row/cores)
    # end = time.time()
    # print("C:\n",result)
    # print("Time Taken:", end - start)