import random
import joblib
from pathlib import Path
from .bpmn_parser import BPMNParser
from .log_analyzer import LogAnalyzer


class BranchPredictor:
    def __init__(self):
        self.probabilities = {}
        self.gateway_branches = {}
        self.gateway_connections = {}

    def fit(self, bpmn_path, log_path):
        parser = BPMNParser(bpmn_path)
        self.gateway_connections = parser.get_gateway_connections()
        self.gateway_branches = {
            gw_id: conn['branches']
            for gw_id, conn in self.gateway_connections.items()
        }

        analyzer = LogAnalyzer(log_path)
        counts = analyzer.count_transitions(self.gateway_connections)
        self.probabilities = analyzer.calculate_probabilities(counts)
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'probabilities': self.probabilities,
            'gateway_branches': self.gateway_branches,
            'gateway_connections': self.gateway_connections
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        predictor = cls()
        predictor.probabilities = data['probabilities']
        predictor.gateway_branches = data['gateway_branches']
        predictor.gateway_connections = data['gateway_connections']
        return predictor

    def predict(self, gateway_id, preceding_activity):
        key = (gateway_id, preceding_activity)
        probs = self.probabilities.get(key)
        
        if not probs:
            branches = self.gateway_branches.get(gateway_id, [])
            if branches:
                return random.choice(branches)
            return None
        
        branches = list(probs.keys())
        weights = list(probs.values())
        return random.choices(branches, weights=weights)[0]

    def get_probabilities(self, gateway_id, preceding_activity):
        key = (gateway_id, preceding_activity)
        return self.probabilities.get(key, {})

