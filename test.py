from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf
import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np
import os
import sys
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

sys.path.insert(1, './utils')
sys.path.insert(1, './initializers')
sys.path.insert(1, './epi_models')
sys.path.insert(1, './scoring_models')
sys.path.insert(1, './edge_estimation_models')
from SparseDistributedMatrix import SparseDistributedMatrix
from SparseDistributedVector import SparseDistributedVector
from Initializer101 import Initializer101
import Simple_SIR as ssir
import ScoringWalker as sw
import StochasticEdgeEstimator as see

sys.path.insert(1, '.')
import SparkDependencyInjection as sdi
import PandiSimConfigInjection as pci
import PandiNetwork as pn
import PandiSim as ps

# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages graphframes:graphframes:0.8.0-spark3.0-s_2.12 pyspark-shell'


spark = SparkSession.builder.master('local')\
    .config(key = "spark.default.parallelism", value = 4)\
    .config(key = "spark.driver.memory", value = "4g")\
    .config(key = "spark.executor.memory", value = "4g")\
    .config(key = "spark.memory.fraction", value = "0.8")\
    .getOrCreate()
# conf=SparkConf.set("spark.default.parallelism", 4)
sc = spark.sparkContext
sc.setCheckpointDir("hdfs://namenode:9000/rddch")
sqlc = SQLContext(sc)

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

sdi.SparkDependencyInjection.set_spark(spark).set_spark_context(sc)
pci.PandiSimConfigInjection.set_write_to("d_pandisim")

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

# u = SparseDistributedVector(sc.parallelize([(0, 1), (1, 2), (2, 3)]), 3)
# v = SparseDistributedVector(sc.parallelize([(0, 1), (1, 2), (2, 3)]), 3)
# a = SparseDistributedMatrix(sc.parallelize([MatrixEntry(0, 1, 1.2),MatrixEntry(1, 0, 2.1),MatrixEntry(0, 2, 4)], 4), 3, 3)

# s = SparseDistributedVector(sc, sc.parallelize([(0, 1), (2, 3)]), 3)
# ones = SparseDistributedVector.repeat(sc, spark, 1, 4)
# twos = SparseDistributedVector.repeat(sc, spark, 2, 4)
# eye = SparseDistributedMatrix.diag(sc, ones)

# print(eye.dot(twos).rdd.collect())

# print(u.dot(a).rdd.collect())
# print(u.dot(a).rdd.collect())
# print(a.dot(a.dot(a.dot(u))).rdd.collect())
# print(v.dot(u))
# print(v.outer(u).entries.collect())
# print(u.op(v).rdd.collect())

# init = Initializer101(6,3)
# init.initialize_vertices()
# init.initialize_edges(init.vertices)
# network = pn.PandiNetwork(init.vertices, init.edges, 6, 3)
# init.vertices.show()
# init.edges.show()
# (truncated, v) = network.verticesToSDV(cond  = (F.col('health_status') != F.lit(-1)))
# print(v.rdd.collect())
# A = network.edgesToSDM(truncated)
# init.edges.show()
# print(A.entries.collect())

# print(e.dot(SparseDistributedMatrix.diag(sc, v)))
# se = v.apply(lambda x: np.exp(-2*x))
# D = SparseDistributedMatrix.diag(sc, se)
# M = A.dot(D)
# C = A.dot(se).apply(lambda x: 1/x).outer(SparseDistributedVector.repeat(sc, spark, 1, A.numRows()))
# P = M.multiply(C)
# print(M.entries.collect())
# print(C.entries.collect())
# r = P.transpose().dot(SparseDistributedVector.repeat(sc, spark, 1/P.numRows(), P.numRows()) )
# r = P.transpose().dot(r)
# r = P.transpose().dot(r)
# r = P.transpose().dot(r)
# r = P.transpose().dot(r)

# ns = v.apply(lambda x: 1 - x).dot(SparseDistributedMatrix.diag(sc, r))

# print(v.rdd.collect())
# print(v.op(ns, 'add').rdd.collect())

sir = ssir.Simple_SIR(
    inits = {'S':0.9, 'I':0.1, 'R':0}, 
    params = {'beta':0.35, 'gamma':0.07, 'N':6, 't_end':20, 'step_size':1}
)
sir.run()
dr = sir.current_sotw()[1]

init = Initializer101(
    nbr_vertices = 6, 
    nbr_edges = 2, 
    nbr_infected = int(dr[0]), 
    nbr_recovered = int(dr[1])
)
# init = Initializer101(
#     nbr_vertices = 20, 
#     nbr_edges = 4, 
#     nbr_infected = 3, 
#     nbr_recovered = 2
# )
init.initialize_vertices()
init.initialize_edges(init.vertices)

# network = pn.PandiNetwork(init.vertices, init.edges, init.nbr_vertices)
network = init.toPandiNetwork()

walker = sw.ScoringWalker(
    network, 
    params = {'alpha-scaler':-2, 'walker-steps':3}
)

# walker.run()
# walker.annotate((2,1))
# network.vertices.show()
# network.edges.show()

edge_est = see.StochasticEdgeEstimator(
    network,
    params = {'SDF': 100, 'alpha': 80, 'beta': 100}
)

# edge_est.run()
# network.vertices.show()
# network.edges.show(50, False)

pandisim = ps.PandiSim(
    network = network, 
    epi_model = sir, 
    scoring_model = walker, 
    edge_model = edge_est, 
    params = {'take_screenshots':False}
)

pandisim.move()
pandisim.take_screenshot()

network.vertices.show()
network.edges.show()