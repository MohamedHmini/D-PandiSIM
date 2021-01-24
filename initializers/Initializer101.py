import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from Initializer import Initializer 

from pyspark.mllib.linalg.distributed import MatrixEntry

import sys
sys.path.insert(1, '../utils')
import SparseDistributedVector as sdv
import SparseDistributedMatrix as sdm


class Initializer101(Initializer):
    def __init__(self, spark, nbr_vertices, nbr_edges, prob_infection = 0.65):
        self.spark = spark
        self.nbr_vertices = nbr_vertices
        self.nbr_edges = nbr_edges
        self.prob_infection = prob_infection
        super().__init__(spark, nbr_vertices, nbr_edges, prob_infection)
    
    def initialize_vertices(self):
        df = self.spark.range(0, self.nbr_vertices, 1).toDF("id")
        df = df.withColumn('score', F.when(F.rand() >= F.lit(self.prob_infection), F.lit(1.0)).otherwise(F.lit(0.0)))
        self.vertices = df.withColumn('health_status', F.col('score'))
        return self.vertices
    
    def initialize_edges(self, vertices):
        src = vertices.select(F.col("id")).orderBy(F.rand()).limit(self.nbr_edges).withColumnRenamed("id", "src") \
        .withColumn("id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        
        src.createOrReplaceTempView("src")
        vertices.createOrReplaceTempView("vertices")
        
        query = self.spark.sql("select vertices.id from vertices minus select src.src from src")
        
        dst = query.orderBy(F.rand()).limit(self.nbr_edges).withColumnRenamed("id", "dst") \
        .withColumn("id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        
        self.edges = src.join(dst, src.id == dst.id).select(F.col('src'), F.col('dst'))
        return self.edges

    # returns the nodes' scores vector
    def verticesToSDV(self, sc, cond):
        df = self.vertices.filter(cond)
        rdd = df.select('id', 'score').rdd.map(lambda row: (row.id, row.score))
        return (df, sdv.SparseDistributedVector(sc, rdd, df.count()))

    # returns the Adjacency matrix
    def edgesToSDM(self, sc, truncated_vertices):
        self.edges.createOrReplaceTempView("edges")
        truncated_vertices.createOrReplaceTempView("vertices")
        real_edges = self.spark.sql("select * from edges where edges.src in (select id from vertices) or edges.dst in (select id from vertices)")
        real_edges.createOrReplaceTempView("real_edges")
        noedge_vertices = self.spark.sql("select * from vertices where vertices.id not in (select real_edges.src from real_edges) and vertices.id not in(select real_edges.dst from real_edges)")
        arti_edges = noedge_vertices.withColumnRenamed("id","src").join(truncated_vertices.select("id").withColumnRenamed("id", "dst")) 
        arti_edges = arti_edges.filter(F.col('src') != F.col('dst'))
        
        
        # src to dst
        entries_1 = real_edges.rdd.map(lambda row: MatrixEntry(row.src, row.dst, 1))
        # dst to src
        entries_2 = real_edges.rdd.map(lambda row: MatrixEntry(row.dst, row.src, 1))
        # self transition 
        entries_3 = truncated_vertices.select("id").rdd.map(lambda row: MatrixEntry(row.id, row.id, 1))
        # edges to avoid self-loop with no uncertainty (randomly distribute the importance of the current node [with artificial edges])
        entries_4 = arti_edges.rdd.map(lambda row: MatrixEntry(row.src, row.dst, 1))

        entries = entries_1.union(entries_2.union(entries_3.union(entries_4)))

        return sdm.SparseDistributedMatrix(sc, entries, self.nbr_vertices, self.nbr_vertices)