from pyspark.mllib.linalg import DenseVector, SparseVector, Vectors, Matrices, SparseMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.mllib.linalg import distributed as D
import SparseDistributedMatrix as sdm




class SparseDistributedVector():

    # default shape of the vector is (N by 1)
    def __init__(self, sc, rdd, size, transposed = False):
        self.sc = sc
        self.rdd = rdd
        self.size = size
        self.transposed = transposed
        
    def _pre_dot(self, A):
        size = self.size
        a = A.entries.map(
            lambda entry: (entry.j, (entry.i, entry.value))
        ).groupByKey() \
        .map(
            lambda x: (x[0], Vectors.sparse(size, *list(zip(*x[1].data))))
        )
        return a
        
    def dot(self, arg):
        if isinstance(arg, sdm.SparseDistributedMatrix):
            return self._dot1(arg)
        elif isinstance(arg, SparseDistributedVector):
            return self._dot2(arg)
    
    def _dot1(self, S):
        if self.size != S.numRows():
            raise Exception(f"size mismatch ({self.size},) and ({S.numRows()},{S.numCols()})")
        sv = self.toSV()
        a = self._pre_dot(S)
        
        c = a.map(
            lambda x: (x[0], x[1].dot(sv))
        )
        return SparseDistributedVector(self.sc, c, S.numCols())
                
    def _dot2(self, v):
        if self.size != v.size:
            raise Exception(f"size mismatch ({self.size},) and ({v.size},)")
        c = self.rdd.union(v.rdd).reduceByKey(
            lambda x,y: x*y
        ).map(lambda x: x[1]).reduce(
            lambda x,y: x+y
        )
        return c
    
    def outer(self, v):
        c = self.rdd.cartesian(v.rdd).map(
            lambda x: MatrixEntry(x[0][0], x[1][0], float(x[0][0]*x[1][0]))
        )
        return sdm.SparseDistributedMatrix(self.sc, c, self.size, v.size)
    
    def op(self, v, op = 'add'):
        if self.size != v.size:
            raise Exception(f"size mismatch ({self.size},) and ({v.size},)")
        c = self.rdd.union(v.rdd).reduceByKey(
            lambda x,y: x+y if op == 'add' else x-y
        )
        return SparseDistributedVector(self.sc, c, self.size)
    
    def repeat(sc, spark, val, size):
        v = spark.range(0,size,1).rdd.map(
            lambda x: (x.id, val)
        )
        return SparseDistributedVector(sc, v, size)

    def toSV(self):
        indices = self.rdd.map(lambda x: x[0]).collect()
        values = self.rdd.map(lambda x: x[1]).collect()
        return Vectors.sparse(self.size, indices, values)