import abc
import sys
sys.path.insert(1, '..')
import SparkDependencyInjection as sdi

class SCORING_model(sdi.SparkDependencyInjection, metaclass=abc.ABCMeta):
    def __init__(self, params, network):
        self.params = params
        self.network = network
        super().__init__()
    
    @abc.abstractmethod
    def run(self):
        pass
    
    @abc.abstractmethod
    def annotate(self):
        pass

    @abc.abstractmethod
    def interact(self):
        pass