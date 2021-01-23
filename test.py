from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np
import os
import sys
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

sys.path.insert(1, './utils')
from SparseDistributedMatrix import SparseDistributedMatrix


# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages graphframes:graphframes:0.8.0-spark3.0-s_2.12 pyspark-shell'


spark = SparkSession.builder.master('local').getOrCreate()
sc = spark.sparkContext
sc.setCheckpointDir("hdfs://namenode:9000/rddch")
sqlc = SQLContext(sc)


a = SparseDistributedMatrix(sc, sc.parallelize([MatrixEntry(0, 0, 1),MatrixEntry(2, 0, 3),MatrixEntry(4, 0, 1)]), 4, 1).transpose()
o = SparseDistributedMatrix.ones(sc, 4).transpose()
r = o.dot(a).entries.collect()

print(r)