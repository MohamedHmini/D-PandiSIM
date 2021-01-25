









class PandiSim():

    def __init__(self, network, epi_model, scoring_model, edge_model, params):
        self.network = network
        self.epi_model = epi_model
        self.scoring_model = scoring_model
        self.edge_model = edge_model
        self.params = params