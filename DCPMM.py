# Libaries used
import numpy as np
import cupy as cp   # CUDA implementation of numpy functionalities (installation requirement for GPU implementation)
import time as time # Used for timing execution
import math         
from scipy.io import mmread # For reading matrix files provided by SuiteSparse Matrix Collection
from multiprocessing import cpu_count # For obtaining the number of cpu cores
from scipy.linalg import blas as FB # Scipy implementation of BLAS dgemm
from pynvml import * # Nvidia library for measuring GPU memory status

"""
Matrix Multiplication
Input: Matrices A, B, and a partitioning threshold c
Output: Resultant Matrix from multiplying A and B
This function first checks to see if the number of rows in A (m)
is less than or equal to the partitioning threshold.

If so, it will perform the matrix multiplication on the GPU using CuPy arrays
and return cupy.matmul(A,B).
Else, the matrices have not met the partitioning threshold and will therefore
be split using the Partition function.
"""
def MatrixMultiply(A,B,c):
    m,n = A.shape
    if m <= c: # Base Case (Partitioning threshold met)
        A = cp.array(A) # Move A to GPU
        B = cp.array(B) # Move B to GPU
        return cp.matmul(A,B) # Perform matrix multiplication on GPU
    return Partition(A,B,c)

"""
Partition
Input: Matrices A, B, and a partitioning threshold c
Output: Resultant Matrix from multiplying A and B
This function first checks to see if the partitioning of A and B can be
performed entirely on the GPU.
This condition checks if there is enough GPU memory to store all the elements of A and B,
which is the dimensions of A and B added times 8 bytes for each element.

If there is enough GPU memory, the partitioning will occur on the GPU.
Else, the partitioning occurs on the CPU.
"""
def Partition(A,B,c):
    m,n = A.shape
    n,p = B.shape
    info = nvmlDeviceGetMemoryInfo(h) # Getting GPU device memory
    if (((m*n) + (n*p)) * 8 < info.free): # Determining if there is enough free GPU memory to store both matrices' elements
        return PartitionGPU(A,B,c)
    else:
        return PartitionCPU(A,B,c)

"""
Partition GPU
Input: Matrices A, B, and a partitioning threshold c
Output: Resultant Matrix from multiplying A and B
This function partitions A and B into smaller blocks, which will then be multiplied
into intermediate results that are combined into the final result.

If A is "wide" (m <= n), A is split in half by its columns and
B is split in half by its rows.

Else if A is "skinny" (m > n), A is split in half by its rows and
B is split in half by its columns.

The submatrix creation is carefully managed to only store what is absolutely necessary
on the GPU and to delete any matrices that no longer need to be used for calculation.
"""
def PartitionGPU(A,B,c):
    m,n = A.shape
    if (type(A) == np.ndarray): # If the arrays are coming from the CPU (numpy arrays),
        A = cp.array(A)         # move them over to the GPU (cupy arrays).
        B = cp.array(B)
    if m <= n: # Case 1 (A is "wide")
        #axis 0 = rows, axis 1 = columns
        [A1,A2] = cp.array_split(A, 2, axis=1) # Split A by columns
        [B1,B2] = cp.array_split(B, 2, axis=0) # Split B by rows
        C1 = cp.array(MatrixMultiply(A1,B1,c)) # Perform A1*B1
        del A1 # A1 no longer needed (remove from GPU)
        del B1 # B1 no longer needed
        C2 = cp.array(MatrixMultiply(A2,B2,c)) # Perform A2*B2
        del A2 # A2 no longer needed
        del B2 # B2 no longer needed
        return C1 + C2 # Return the final result C = C1 + C2 = A1*B1 + A2*B2
    else: #m > n Case 2 (A is "skinny")
        [A1,A2] = cp.array_split(A, 2, axis=0) # Split A by rows
        [B1,B2] = cp.array_split(B, 2, axis=1) # Split B by columns

        C1 = MatrixMultiply(A1,B1,c) # Perform A1*B1
        C2 = MatrixMultiply(A1,B2,c) # Perform A1*B2
        C12 = cp.hstack((C1,C2))     # Stack C1 and C2 horizontally -> C12
        del C1  # C1 no longer needed
        del C2  # C2 no longer needed

        C3 = MatrixMultiply(A2,B1,c) # Perform A2*B1
        C4 = MatrixMultiply(A2,B2,c) # Perform A2*B2
        C34 = cp.hstack((C3,C4))     # Stack C3 and C4 horizontally -> C34
        del C3  # C3 no longer needed
        del C4  # C4 no longer needed
        return cp.vstack((C12,C34))  # Stack C12 and C34 vertically to form final result C
        
"""
Partition CPU
Input: Matrices A, B, and a partitioning threshold c
Output: Resultant Matrix from multiplying A and B
This function partitions A and B into smaller blocks, which will then be multiplied
into intermediate results that are combined into the final result.

If A is "wide" (m <= n), A is split in half by its columns and
B is split in half by its rows.

Else if A is "skinny" (m > n), A is split in half by its rows and
B is split in half by its columns.
"""
def PartitionCPU(A,B,c):
    m,n = A.shape
    if (type(A) == cp.ndarray): # If the arrays are coming from the GPU (cupy arrays),
        A = cp.asnumpy(A)       # move them over to the CPU (numpy arrays).
        B = cp.asnumpy(B)
    # Everything is the same as PartitionGPU with numpy functions instead of cupy functions.
    # Also, intermediate results of matrix multiplication moved to CPU so no GPU memory management needed.
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

# Numpy implementation of Matrix Multiplication
# Uses BLAS level 3 routine dgemm under the hood
def numpyMult(A,B):
    return np.dot(A,B)

if __name__ == "__main__":
    nvmlInit() # Initializing GPU memory tracker
    h = nvmlDeviceGetHandleByIndex(0) # GPU device labeled h
    info = nvmlDeviceGetMemoryInfo(h) # Get info from GPU

    # Establishing the partitioning threshold
    # The threshold should be such that we can store both A and B on the GPU to perform matrix multiplication.
    # The maximum number of elements would therefore be the amount of free memory / 8 bytes per element.
    # The max number of rows would be the square root of the number of elements assuming a square matrix.
    # Both A and B must be stored so the threshold for A would be the number of rows divided by 2.
    partitionLimit = math.sqrt(info.free / (8)) / 2

    # Running the algorithm on 5 matrices from the SSMC and 5 dense matrices of the same dimensions
    d = {} # Dictionary for storing execution time of each dataset
    fileNames = ['datasets/494_bus.mtx', 'datasets/bcsstk17/bcsstk17.mtx', 'datasets/ex11/ex11.mtx', 'datasets/gupta3/gupta3.mtx', 'human_gene1/human_gene1.mtx', 'human_gene2/human_gene2.mtx']
    for fileName in fileNames: # Initialize times to 0
        d[fileName] = 0
        d[fileName + 'Dense'] = 0
        #d[fileName+'Numpy'] = 0
        #d[fileName+'NumpyDense'] = 0
    for i in range(10): # For 10 iterations (totaling time for 10 and then averaging)
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
            result = MatrixMultiply(A,B,partitionLimit) # Tracking execution time for our algorithm on the sparse dataset
            end = time.time()
            print("Time Taken:", end - start)
            d[fileName] += end - start # Adding to time dictionary

            # Numpy executions on sparse datasets (commented out because of long execution times)
            # start = time.time()
            # result = numpyMult(A, B)
            # end = time.time()
            #print("Time Taken:", end - start)
            #d[fileName+'Numpy'] += end - start

            print("Dense:")
            A = np.random.randn(row, col)
            A = np.asarray(np.float32(A))
            B = np.asarray(np.float32(A))
            start = time.time()
            result = MatrixMultiply(A,B,partitionLimit) # Tracking execution time for our algorithm on the dense dataset
            end = time.time()
            print("Time Taken:", end - start)
            d[fileName + 'Dense'] += end - start

            # Numpy executions on dense datasets (commented out because of long execution times)
            # start = time.time()
            # result = numpyMult(A, B)
            # end = time.time()
            #print("Time Taken:", end - start)
            #d[fileName+'NumpyDense'] += end - start

        print("Current Total:", d)
    # Averaging execution time across all 10 runs
    for fileName in fileNames:
        d[fileName] /= 10.0
        d[fileName + 'Dense'] /= 10.0
        #d[fileName+'Numpy'] /= 10.0
        #d[fileName+'NumpyDense'] /= 10.0
    print("Avg (10 iterations):", d)
    # print()
    # import pandas as pd
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # data = pd.DataFrame(d, index=[0])
    # print(data)