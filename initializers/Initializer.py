import abc




class Initializer(metaclass = abc.ABCMeta):

    def __init__(self, spark, nbr_vertices, nbr_edges, prob_infection = 0.65):
        self.spark = spark
        self.nbr_vertices = nbr_vertices
        self.nbr_edges = nbr_edges
        self.prob_infection = prob_infection
        super().__init__()

    @abc.abstractmethod
    def initialize_vertices(self):
        pass
    
    @abc.abstractmethod
    def initialize_edges(self, vertices):
        pass