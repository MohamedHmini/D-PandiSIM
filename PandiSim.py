
from pyspark import StorageLevel

import sys
import os
sys.path.insert(1, './utils')

import PandiSimConfigInjection as config
import SparkDependencyInjection as sdi




class PandiSim(sdi.SparkDependencyInjection, config.PandiSimConfigInjection):

    def __init__(self, network, epi_model, scoring_model, edge_model, params = {'take_screenshots':False, 'destroy':False}):
        self.network = network
        self.epi_model = epi_model
        self.scoring_model = scoring_model
        self.edge_model = edge_model
        self.params = params
        # self.params['t_end'] = epi_model.params['t_end']

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
            if self.epi_model.step >= 2 and self.params['destroy']:
                self.read_state()
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

    def read_state(self):
        hdfs = "hdfs://namenode:9000/"
        edges_fil = os.path.join(hdfs, self.read_from, f"step_{self.epi_model.step}", "edges.csv")
        vertices_fil = os.path.join(hdfs, self.read_from, f"step_{self.epi_model.step}", "vertices.csv")
        self.network.vertices = self.spark.read.format("csv").option("delimiter", ',')\
            .option('header', False).option('inferSchema', True).load(vertices_fil).toDF('id', 'score', 'health_status')\
            .sort('id').cache()
        self.network.edges = self.spark.read.format("csv").option("delimiter", ',')\
            .option('header', False).option('inferSchema', True).load(edges_fil).toDF('src', 'dst')\
            .cache()