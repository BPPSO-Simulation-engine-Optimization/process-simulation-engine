import pm4py
from collections import defaultdict


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

    def count_transitions(self, gateway_connections):
        activity_to_gateways = {}
        for gw_id, conn in gateway_connections.items():
            for act in conn['preceding']:
                if act not in activity_to_gateways:
                    activity_to_gateways[act] = []
                activity_to_gateways[act].append({
                    'gateway': gw_id,
                    'branches': conn['branches']
                })

        counts = defaultdict(lambda: defaultdict(int))
        for activities in self.iter_traces():
            for i in range(len(activities) - 1):
                current = activities[i]
                next_act = activities[i + 1]
                if current not in activity_to_gateways:
                    continue
                for gw_info in activity_to_gateways[current]:
                    if next_act in gw_info['branches']:
                        key = (gw_info['gateway'], current)
                        counts[key][next_act] += 1

        return dict(counts)

    def calculate_probabilities(self, transition_counts):
        probabilities = {}
        for key, branch_counts in transition_counts.items():
            total = sum(branch_counts.values())
            if total > 0:
                probabilities[key] = {
                    branch: count / total
                    for branch, count in branch_counts.items()
                }
        return probabilities







