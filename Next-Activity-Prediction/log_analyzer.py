import pm4py


class LogAnalyzer:
    def __init__(self, log_path):
        self.log = pm4py.read_xes(log_path)
        if hasattr(self.log, 'iterrows'):
            self.log = pm4py.convert_to_event_log(self.log)

    def get_trace_activities(self, trace):
        activities = []
        for event in trace:
            if 'concept:name' in event:
                activities.append(event['concept:name'])
        return activities

    def iter_traces(self):
        for trace in self.log:
            yield self.get_trace_activities(trace)

