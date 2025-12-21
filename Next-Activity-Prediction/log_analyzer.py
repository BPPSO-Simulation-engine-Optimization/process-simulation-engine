import pm4py


class LogAnalyzer:
    def __init__(self, log_path):
        self.log = pm4py.read_xes(log_path)
        if hasattr(self.log, 'iterrows'):
            self.log = pm4py.convert_to_event_log(self.log)

