from pyspark.mllib.linalg import DenseVector, SparseVector, Vectors, Matrices, SparseMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.mllib.linalg import distributed as D
import SparseDistributedVector as sdv

import sys
import SparkDependencyInjection as sdi


class SparseDistributedMatrix(sdi.SparkDependencyInjection):
    def __init__(self, entries, numRows=0, numCols=0):
        self.entries = entries
        self.rows = numRows
        self.cols = numCols
        super(SparseDistributedMatrix, self).__init__()

    def numRows(self):
        return self.rows
    def numCols(self):
        return self.cols
    
    def transpose(self):
        entries = self.entries.map(
            lambda entry: MatrixEntry(entry.j, entry.i, entry.value)
        )
        return SparseDistributedMatrix(entries, self.numCols(), self.numRows())
    
    def _pre_dot(self, A, by = 'row', size = 0):
        a = A.entries.map(
            lambda entry: (entry.i, (entry.j, entry.value)) if by == 'row' else (entry.j, (entry.i, entry.value))
        ).groupByKey() \
        .map(
            lambda x: (x[0], Vectors.sparse(size, *list(zip(*sorted(x[1].data, key = lambda x: x[0])))))
        )
        return a

    def dot(self, arg):
        if isinstance(arg,SparseDistributedMatrix):
            return self._dot1(arg)
        elif type(arg) == sdv.SparseDistributedVector:
            return self._dot2(arg)

    def _dot1(self, B):
        if self.numCols() != B.numRows():
            raise Exception(f"size mismatch {(self.numRows(), self.numCols())}, {(B.numRows(), B.numCols())}")
        a = self._pre_dot(self, 'row')
        b = self._pre_dot(B, 'col')
        
        
        c = a.cartesian(b).map(
            lambda x: MatrixEntry(x[0][0], x[1][0], x[0][1].dot(x[1][1]))
        ).filter(
            lambda entry: entry.value != 0.0
        )
            
        return SparseDistributedMatrix(c, self.numRows(), B.numCols())

    def _dot2(self,v):
        if self.numCols() != v.size:
            raise Exception(f"size mismatch ({self.numRows()},{self.numCols()}) and ({v.size},)")
        size = v.size
        sv = v.rdd.map(lambda entry: (1, entry)).groupByKey().map(
            lambda x: (x[0], Vectors.sparse(size, *list(zip(*sorted(x[1].data, key = lambda x: x[0])))))
        )
        a = self._pre_dot(self, size = size)

        c = sv.cartesian(a).map(
            lambda x: (x[1][0], float(x[0][1].dot(x[1][1])))
        ).filter(
            lambda entry: entry[1] != 0.0
        )
        return sdv.SparseDistributedVector(c, self.numRows())

    
    def diag(vect):
        c = vect.rdd.map(
            lambda entry : MatrixEntry(entry[0], entry[0], entry[1])
        ) 
        return SparseDistributedMatrix(c, vect.size, vect.size)
    
    # def ones(sc, size:int):
    #     c = SparseDistributedMatrix(sc, sc.parallelize([MatrixEntry(0,i,1) for i in range(size)]), 1, size)
    #     return c
    
    def size(self):
        return (self.numRows(), self.numCols())
    
    def fromDF(self, df):
        pass

    def toDF(self):
        pass

    def _pre_arithmetic_op(self, A, B):
        a = self.entries.map(
            lambda entry: ((entry.i,entry.j),entry.value)
        )
        b = B.entries.map(
            lambda entry: ((entry.i,entry.j),entry.value)
        )
        return a,b
        
    
    def multiply(self, B):
        a,b = self._pre_arithmetic_op(self, B)
        c = a.union(b).groupByKey().map(
            lambda x : MatrixEntry(x[0][0],x[0][1], x[1].data[0] * x[1].data[1] if len(x[1].data) == 2 else 0) 
        )
        
        return SparseDistributedMatrix(c, self.numRows(), self.numCols())
    
    # def multiply(self, b:float):
    #     c = self.entries.map(
    #         lambda entry : MatrixEntry(entry.i, entry.j, entry.value * b) 
    #     )
        
    #     return SparseDistributedMatrix(self.sc, c, self.numRows(), self.numCols())