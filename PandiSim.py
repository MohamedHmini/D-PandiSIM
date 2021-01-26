


import sys
sys.path.insert(1, './utils')

import PandiSimConfigInjection as config




class PandiSim(config.PandiSimConfigInjection):

    def __init__(self, network, epi_model, scoring_model, edge_model, params):
        self.network = network
        self.epi_model = epi_model
        self.scoring_model = scoring_model
        self.edge_model = edge_model
        self.params = params

    def move(self):
        sotw = self.epi_model.next_sotw()[1]
        self.scoring_model.run()
        self.scoring_model.annotate(sotw)
        self.network.vertices.show()
        # self.edge_model.run()


    def run(self):
        pass
