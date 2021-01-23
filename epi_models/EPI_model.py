import abc




class EPI_model(metaclass=abc.ABCMeta):
    step = 0
    all_sotw = []
    def __init__(self, name, inits, params):
        self.name = name
        self.inits = inits
        self.params = params
        super().__init__()
        
    def re_init(self, inits, params):
        self.inits = inits
        self.params = params
        self.step = 0
        self.all_sotw = []
        
    # needs pre and post wrappers
    @abc.abstractmethod
    def run(self):
        pass
    
    @abc.abstractmethod
    def current_sotw(self):
        pass
    
    @abc.abstractmethod
    def next_sotw(self):
        pass
    
    @abc.abstractmethod
    def previous_sotw(self):
        pass
    
    @abc.abstractmethod
    def interact(self):
        pass