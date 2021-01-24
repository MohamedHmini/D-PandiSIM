





class SparkDependencyInjection(object):
    def __init__(self):
        super(SparkDependencyInjection, self).__init__()
    def set_spark(spark):
        SparkDependencyInjection.spark = spark
        return SparkDependencyInjection
    def set_spark_context(sc):
        SparkDependencyInjection.sc = sc
        return SparkDependencyInjection