import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql.window import Window

import sys
sys.path.insert(1, '../utils')
import SparkDependencyInjection as sdi
import EDGE_estimator as ee


class StochasticEdgeEstimator(ee.EDGE_estimator):
    def __init__(self, network, params = {'SDF': 10, 'alpha': 10, 'beta': 10}):
        super().__init__(network, params)
        
    def run(self):
        diff1 = self.network.vertices.select('id').exceptAll(self.network.edges.select('src'))
        diff2 = self.network.vertices.select('id').exceptAll(self.network.edges.select('dst'))
        diff = diff1.intersect(diff2).withColumn('row', F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        src = diff.alias('src').withColumnRenamed('id', 'src')
        dst = diff.alias('dst').withColumnRenamed('id', 'dst')
        noedges = src.join(dst, F.col("src.row") < F.col("dst.row")).select(src.src, dst.dst)
        a,b = self.params['alpha'], self.params['beta'] + self.params['SDF']
        beta = F.udf(lambda x: float(np.random.binomial(1, np.random.beta(a,b))))
        newedges = noedges.withColumn("edge", beta(F.col('src'))).filter("edge == 1.0").select('src', 'dst')
    
        self.network.edges = newedges.union(self.network.edges)

    def annotate(self):
        pass

    def interact(self):
        pass