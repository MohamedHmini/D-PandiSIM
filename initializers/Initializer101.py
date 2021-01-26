import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from Initializer import Initializer 
from pyspark import StorageLevel


from pyspark.mllib.linalg.distributed import MatrixEntry

import sys
sys.path.insert(1, '../utils')
sys.path.insert(1, '..')
import SparseDistributedVector as sdv
import SparseDistributedMatrix as sdm
import PandiNetwork as pn


class Initializer101(Initializer):
    def __init__(self, nbr_vertices, nbr_edges, nbr_infected = 0, nbr_recovered = 0):
        self.nbr_vertices = nbr_vertices
        self.nbr_edges = nbr_edges
        self.nbr_infected = nbr_infected
        self.nbr_recovered = nbr_recovered
        super().__init__(nbr_vertices, nbr_edges)
    
    def initialize_vertices(self):
        df = self.spark.range(0, self.nbr_vertices, 1).toDF("id").orderBy(F.rand()).persist(StorageLevel.MEMORY_AND_DISK)
        # df = df.withColumn('score', F.when(F.rand() >= F.lit(self.prob_infection), F.lit(1.0)).otherwise(F.lit(0.0)))
        infected = df.limit(self.nbr_infected).withColumn('score', F.lit(1.0))
        recovered = df.select('id').exceptAll(infected.select('id')).limit(self.nbr_recovered)
        recovered = recovered.withColumn('score', F.lit(-1.0))
        total = infected.union(recovered).persist(StorageLevel.MEMORY_AND_DISK)
        rest = df.select('id').exceptAll(total.select('id')).withColumn('score', F.lit(0.0))
        self.vertices = rest.union(total).withColumn('health_status', F.col('score')).orderBy('id').persist(StorageLevel.MEMORY_AND_DISK)
        return self.vertices
    
    def initialize_edges(self, vertices):
        src = vertices.select(F.col("id")).orderBy(F.rand()).limit(self.nbr_edges).withColumnRenamed("id", "src") \
        .withColumn("id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        
        src.createOrReplaceTempView("src")
        vertices.createOrReplaceTempView("vertices")
        
        query = self.spark.sql("select vertices.id from vertices minus select src.src from src")
        
        dst = query.orderBy(F.rand()).limit(self.nbr_edges).withColumnRenamed("id", "dst") \
        .withColumn("id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        
        self.edges = src.join(dst, src.id == dst.id).select(F.col('src'), F.col('dst')).persist(StorageLevel.MEMORY_AND_DISK)
        return self.edges

    def toPandiNetwork(self):
        return pn.PandiNetwork(self.vertices, self.edges, self.nbr_vertices)