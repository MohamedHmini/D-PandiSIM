
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark import StorageLevel


from pyspark.mllib.linalg.distributed import MatrixEntry


import sys
sys.path.insert(1, './utils')
sys.path.insert(1, '.')
import SparseDistributedVector as sdv
import SparseDistributedMatrix as sdm
import SparkDependencyInjection as sdi


class PandiNetwork(sdi.SparkDependencyInjection):
    def __init__(self, vertices, edges, nbr_vertices):
        self.vertices = vertices
        self.edges = edges
        self.nbr_vertices = nbr_vertices

    def get_vertices_schema(self):
        vertices_schema = T.StructType(
            [
                T.StructField(name = "id", dataType=T.IntegerType(), nullable = False),
                T.StructField(name = "score", dataType=T.FloatType(), nullable = False),
                T.StructField(name = "health_status", dataType=T.IntegerType(), nullable = False),
            ]
        )
        return vertices_schema

    def toVertices(self, sdv):
        return sdv.rdd.toDF(['id', 'score'])

    # returns the nodes' scores vector
    def verticesToSDV(self, cond):
        df = self.vertices.filter(cond).persist(StorageLevel.MEMORY_AND_DISK)
        rdd = df.select('id', 'score').rdd.map(lambda row: (row.id, row.score)).persist(StorageLevel.MEMORY_AND_DISK)
        self.vertices_sdv = sdv.SparseDistributedVector(rdd, df.count())
        return (df, self.vertices_sdv)

    # returns the Adjacency matrix
    def edgesToSDM(self, truncated_vertices):
        self.edges.createOrReplaceTempView("edges")
        truncated_vertices.createOrReplaceTempView("vertices")
        real_edges = self.spark.sql("select * from edges where edges.src in (select id from vertices) or edges.dst in (select id from vertices)").persist(StorageLevel.MEMORY_AND_DISK)
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

        entries = entries_1.union(entries_2.union(entries_3.union(entries_4))).persist(StorageLevel.MEMORY_AND_DISK)
        size = truncated_vertices.count()
        self.edges_sdm = sdm.SparseDistributedMatrix(entries, size, size)
        return self.edges_sdm
