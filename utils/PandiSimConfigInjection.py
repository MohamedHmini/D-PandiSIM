





class PandiSimConfigInjection(object):
    def __init__(self):
        super(PandiSimConfigInjection, self).__init__()
    def set_is_interactive(c = False):
        PandiSimConfigInjection.isInteractive = c
        return PandiSimConfigInjection
    def set_read_from(path):
        PandiSimConfigInjection.read = path
        return PandiSimConfigInjection
    def set_write_to(path):
        PandiSimConfigInjection.write = path
        return PandiSimConfigInjection