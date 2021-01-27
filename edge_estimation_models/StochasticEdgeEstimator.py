import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np
import scipy
from pyspark.sql.window import Window
from pyspark import StorageLevel

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import scipy.integrate as spi
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets 
from ipywidgets import interact

import sys
sys.path.insert(1, '../utils')
import SparkDependencyInjection as sdi
import EDGE_estimator as ee


class StochasticEdgeEstimator(ee.EDGE_estimator):
    def __init__(self, network, params = {'SDF': 10, 'alpha': 10, 'beta': 10}):
        super().__init__("Stochastic Edge Estimator", network, params)
        
    def run(self):
        # diff1 = self.network.vertices.select('id').exceptAll(self.network.edges.select('src'))
        # diff2 = self.network.vertices.select('id').exceptAll(self.network.edges.select('dst'))
        # diff = diff1.union(diff2).withColumn('row', F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))).persist(StorageLevel.MEMORY_AND_DISK)
        sth = self.network.vertices.withColumn('row', F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))).persist(StorageLevel.MEMORY_AND_DISK)
        src = sth.alias('src').withColumnRenamed('id', 'src')
        dst = sth.alias('dst').withColumnRenamed('id', 'dst')
        noedges = src.join(dst, F.col("src.row") < F.col("dst.row")).select(src.src, dst.dst)
        noedges = noedges.exceptAll(self.network.edges)
        a,b = self.params['alpha'], self.params['beta'] + self.params['SDF']
        beta = F.udf(lambda x: float(np.random.binomial(1, np.random.beta(a,b))))
        newedges = noedges.withColumn("edge", beta(F.col('src'))).filter("edge == 1.0").select('src', 'dst')
    
        self.network.edges = newedges.unionAll(self.network.edges).persist(StorageLevel.MEMORY_AND_DISK)

    def annotate(self):
        pass

    def _interact(self, alpha, beta, SDF):
        self.params['alpha'], self.params['beta'], self.params['SDF'] = alpha, beta, SDF
        x = np.linspace(0, 1, 200)
        y = scipy.stats.beta.pdf(x, alpha, beta + SDF)
        plt.figure(figsize = (15,10))
        plt.plot(x,y, label = 'beat density function')
        plt.title("the used PDF to stochastically assign new edges")
        plt.legend()
    
    def interact(self):
        interact(self._interact, alpha=(0, 200, 5), beta = (0, 100, 5), SDF = (0, 100, 5))
