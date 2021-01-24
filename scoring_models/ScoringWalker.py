
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
        self.sotw_scores = v.apply(lambda x: 1 - x).dot(sdm.SparseDistributedMatrix.diag(r))

    
    def annotate(self):
        pass

    def interact(self):
        pass