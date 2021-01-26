
import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np
from pyspark import StorageLevel

import SCORING_model as sm

import sys
sys.path.insert(1, '../utils')
import SparseDistributedMatrix as sdm
import SparseDistributedVector as sdv



class ScoringWalker(sm.SCORING_model):
    def __init__(self, network, params = {'alpha-scaler':-2, 'walker-steps':10}):
        super().__init__(params, network)
    
    def run(self):
        (truncated, v) = self.network.verticesToSDV(cond  = (F.col('health_status') != F.lit(-1)))
        A = self.network.edgesToSDM(truncated)
        scaler = self.params['alpha-scaler']
        se = v.apply(lambda x: np.exp(scaler*x))
        se.rdd = se.rdd.persist(StorageLevel.MEMORY_AND_DISK)
        D = sdm.SparseDistributedMatrix.diag(se) 
        M = A.dot(D)
        C = A.dot(se).apply(lambda x: 1/x).outer(sdv.SparseDistributedVector.repeat(1, A.numRows()))
        P = M.multiply(C).transpose()
        P.entries = P.entries.persist(StorageLevel.MEMORY_AND_DISK)

        # running the walker:
        r = sdv.SparseDistributedVector.repeat(1/P.numRows(), P.numRows()) 
        for _ in range(self.params['walker-steps']):
            r = P.dot(r)

        # # updating the scores
        self.sotw_scores = v.op(v.apply(lambda x: 1 - x).dot(sdm.SparseDistributedMatrix.diag(r)), 'add')

    
    def annotate(self, sotw = (0,0)):
        nbr_infected = sotw[0]
        nbr_recovered = sotw[1]

        vertices = self.network.toVertices(self.sotw_scores)
        new_stuff = self.network.vertices.select("id", "health_status").join(vertices, on=['id']).persist(StorageLevel.MEMORY_AND_DISK)

        df = new_stuff.filter((F.col('health_status') != F.lit(1.0)) & (F.col('health_status') != F.lit(-1.0)))
        if nbr_infected != 0:
            infected = df.sort(F.col('score').desc()).limit(nbr_infected).persist(StorageLevel.MEMORY_AND_DISK)
            infected = infected.withColumn('health_status', F.lit(1.0)).withColumn('score', F.lit(1.0))
        else:
            infected = self.spark.createDataFrame(self.sc.emptyRDD(), self.network.get_vertices_schema())
        
        if nbr_recovered != 0:
            recovered = new_stuff.filter(F.col('health_status') == F.lit(1.0)).orderBy(F.rand()).limit(nbr_recovered)
            recovered = recovered.withColumn('health_status', F.lit(-1.0)).withColumn('score', F.lit(0.0)).persist(StorageLevel.MEMORY_AND_DISK)
        else:
            recovered = self.spark.createDataFrame(self.sc.emptyRDD(), self.network.get_vertices_schema())
        

        annotated = infected.union(recovered).persist(StorageLevel.MEMORY_AND_DISK)

        diff = new_stuff.select(F.col('id')).exceptAll(annotated.select(F.col('id')))
        rest = new_stuff.join(diff, on = ['id'], how = 'inner')

        results = rest.union(annotated).select('id', 'score', 'health_status').persist(StorageLevel.MEMORY_AND_DISK)
        diff = self.network.vertices.select(F.col('id')).exceptAll(results.select(F.col('id')))
        rest = self.network.vertices.join(diff, on = ['id'], how = 'inner').select('id', 'score', 'health_status')
        self.network.vertices = rest.union(results).orderBy('id').persist(StorageLevel.MEMORY_AND_DISK)

        # # remove the recovered vertices' edges
        recovered_src = self.network.edges.select('src').exceptAll(recovered.select('id')).join(self.network.edges, on =['src'])
        recovered_dst = self.network.edges.select('dst').exceptAll(recovered.select('id')).join(self.network.edges, on =['dst']).select('src', 'dst')
        self.network.edges = recovered_src.intersect(recovered_dst.select('src', 'dst')).persist(StorageLevel.MEMORY_AND_DISK)


    def interact(self):
        # networkx visualization
        pass