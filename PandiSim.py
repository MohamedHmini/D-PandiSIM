
import sys
import os
sys.path.insert(1, './utils')

import PandiSimConfigInjection as config




class PandiSim(config.PandiSimConfigInjection):

    def __init__(self, network, epi_model, scoring_model, edge_model, params = {'take_screenshots':False}):
        self.network = network
        self.epi_model = epi_model
        self.scoring_model = scoring_model
        self.edge_model = edge_model
        self.params = params
        self.params['t_end'] = epi_model.params['t_end']

    def move(self):
        sotw = self.epi_model.next_sotw()[1]
        self.scoring_model.run()
        self.scoring_model.annotate(sotw)
        self.edge_model.run()

    def _perc_to_steps(self, perc):
        return int(perc * self.params['t_end'])

    def run(self, perc = 0.1):
        stopAt = self._perc_to_steps(perc)

        for _ in range(stopAt):
            self.move()
            self.take_screenshot()

    def take_screenshot(self):
        hdfs = "hdfs://namenode:9000/"
        edges_fil = os.path.join(hdfs, self.write_to, f"step_{self.epi_model.step}", "edges.csv")
        vertices_fil = os.path.join(hdfs, self.write_to, f"step_{self.epi_model.step}", "vertices.csv")
        print(edges_fil)
        print(vertices_fil)
        if self.params['take_screenshots']:
            self.network.vertices\
                .write.format("csv").option("delimiter", ',')\
                .option('header', False).mode('overwrite').save(vertices_fil)
            self.network.edges\
                .write.format("csv").option("delimiter", ',')\
                .option('header', False).mode('overwrite').save(edges_fil)