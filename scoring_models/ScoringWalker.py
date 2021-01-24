
import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np

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
        D = sdm.SparseDistributedMatrix.diag(se)
        M = A.dot(D)
        C = A.dot(se).apply(lambda x: 1/x).outer(sdv.SparseDistributedVector.repeat(1, A.numRows()))
        P = M.multiply(C)

        # running the walker:
        r = sdv.SparseDistributedVector.repeat(1/P.numRows(), P.numRows()) 
        for _ in range(self.params['walker-steps']):
            r = P.transpose().dot(r)

        # updating the scores
        self.sotw_scores = v.op(v.apply(lambda x: 1 - x).dot(sdm.SparseDistributedMatrix.diag(r)), 'add')

    
    def annotate(self, sotw = {'I':0.0,'R':0.0}, previous_sotw = {'I':0.0,'R':0.0}):
        nbr_infected = (sotw['I'] - previous_sotw['I'])*self.network.nbr_vertices if sotw['I'] > previous_sotw['I'] else 0
        nbr_recovered = (sotw['R'] - previous_sotw['R'])*self.network.nbr_vertices if sotw['R'] > previous_sotw['R'] else 0

        vertices = self.network.toVertices(self.sotw_scores)
        new_stuff = self.network.vertices.select("id", "health_status").join(vertices, on=['id'])

        df = new_stuff.filter((F.col('health_status') != F.lit(1.0)) & (F.col('health_status') != F.lit(-1.0)))
        infected = df.sort(F.col('score').desc()).limit(nbr_infected)
        infected = infected.withColumn('health_status', F.lit(1.0)).withColumn('score', F.lit(1.0))
        
        recovered = new_stuff.filter(F.col('health_status') == F.lit(1.0)).orderBy(F.rand()).limit(nbr_recovered)
        recovered = recovered.withColumn('health_status', F.lit(-1.0)).withColumn('score', F.lit(0.0))

        annotated = infected.union(recovered)
        diff = new_stuff.select(F.col('id')).exceptAll(annotated.select(F.col('id')))
        rest = new_stuff.join(diff, on = ['id'], how = 'inner')

        results = rest.union(annotated)

        diff = self.network.vertices.select(F.col('id')).exceptAll(results.select(F.col('id')))
        rest = self.network.vertices.join(diff, on = ['id'], how = 'inner')
        
        self.network.vertices = rest.union(results)


    def interact(self):
        # networkx visualization
        pass