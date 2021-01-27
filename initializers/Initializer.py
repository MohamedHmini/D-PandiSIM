import abc

import sys
sys.path.insert(1, '../utils')
import SparkDependencyInjection as sdi


class Initializer(sdi.SparkDependencyInjection, metaclass = abc.ABCMeta):

    def __init__(self, nbr_vertices, nbr_edges, nbr_infected = 0, nbr_recovered = 0):
        self.nbr_vertices = nbr_vertices
        self.nbr_edges = nbr_edges
        self.nbr_infected = nbr_infected
        self.nbr_recovered = nbr_recovered
        super().__init__()

    @abc.abstractmethod
    def initialize_vertices(self):
        pass
    
    @abc.abstractmethod
    def initialize_edges(self, vertices):
        pass
    
    @abc.abstractmethod
    def toPandiNetwork(self):
        pass