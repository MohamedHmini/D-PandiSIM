from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np
import os
import sys
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

sys.path.insert(1, './utils')
sys.path.insert(1, './initializers')
sys.path.insert(1, './epi_models')
from SparseDistributedMatrix import SparseDistributedMatrix
from SparseDistributedVector import SparseDistributedVector
from Initializer101 import Initializer101
from Simple_SIR import Simple_SIR



# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages graphframes:graphframes:0.8.0-spark3.0-s_2.12 pyspark-shell'


spark = SparkSession.builder.master('local').getOrCreate()
sc = spark.sparkContext
sc.setCheckpointDir("hdfs://namenode:9000/rddch")
sqlc = SQLContext(sc)


# a = SparseDistributedMatrix(sc, sc.parallelize([MatrixEntry(0, 0, 1),MatrixEntry(2, 0, 3),MatrixEntry(4, 0, 1)]), 4, 1).transpose()
# o = SparseDistributedMatrix.ones(sc, 4).transpose()
# r = o.dot(a).entries.collect()
# print(r)

# init = Initializer101(spark, 20,5)
# df = init.initialize_vertices()
# df.show()
# init.initialize_edges(df).show()


# sir = Simple_SIR()
# sir.run()
# print(sir.next_sotw())

u = SparseDistributedVector(sc, sc.parallelize([(0, 1), (1, 2), (2, 3)]), 3)
v = SparseDistributedVector(sc, sc.parallelize([(0, 1), (1, 2), (2, 3)]), 3)
a = SparseDistributedMatrix(sc, sc.parallelize([MatrixEntry(0, 1, 1.2),MatrixEntry(1, 0, 2.1),MatrixEntry(0, 2, 4)], 4), 3, 3)

s = SparseDistributedVector(sc, sc.parallelize([(0, 1), (2, 3)]), 3)
ones = SparseDistributedVector.repeat(sc, spark, 1, 4)
twos = SparseDistributedVector.repeat(sc, spark, 2, 4)
eye = SparseDistributedMatrix.diag(sc, ones)

print(eye.dot(twos).rdd.collect())


# print(u.dot(a).rdd.collect())
# print(a.dot(u).rdd.collect())
# print(v.dot(u))
# print(v.outer(u).entries.collect())
# print(u.op(v).rdd.collect())

