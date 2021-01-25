import abc

import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np

import sys
sys.path.insert(1, '../utils')
import SparkDependencyInjection as sdi



class EDGE_estimator(sdi.SparkDependencyInjection, metaclass=abc.ABCMeta):
    def __init__(self, network, params):
        self.network = network
        self.params = params
        super().__init__()
        
    @abc.abstractmethod
    def run(self):
        pass
    
    @abc.abstractmethod
    def interact(self):
        pass