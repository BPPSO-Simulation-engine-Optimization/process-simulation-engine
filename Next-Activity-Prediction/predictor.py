import random


class BranchPredictor:
    def __init__(self):
        self.probabilities = {}
        self.gateway_branches = {}

    def set_probabilities(self, probabilities, gateway_connections):
        self.probabilities = probabilities
        self.gateway_branches = {
            gw_id: conn['branches']
            for gw_id, conn in gateway_connections.items()
        }

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

