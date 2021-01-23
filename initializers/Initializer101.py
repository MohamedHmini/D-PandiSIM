import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from Initializer import Initializer 

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
        df = df.withColumn('health_status', F.col('score'))
        return df
    
    def initialize_edges(self, vertices):
        src = vertices.select(F.col("id")).orderBy(F.rand()).limit(self.nbr_edges).withColumnRenamed("id", "src") \
        .withColumn("id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        
        src.createOrReplaceTempView("src")
        vertices.createOrReplaceTempView("vertices")
        
        query = self.spark.sql("select vertices.id from vertices minus select src.src from src")
        
        dst = query.orderBy(F.rand()).limit(self.nbr_edges).withColumnRenamed("id", "dst") \
        .withColumn("id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        
        return src.join(dst, src.id == dst.id).select(F.col('src'), F.col('dst'))