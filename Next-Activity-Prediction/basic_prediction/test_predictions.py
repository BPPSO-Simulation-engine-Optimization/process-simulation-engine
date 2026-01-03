"""
Test the branch predictor - demonstrates loading a pre-trained model.
"""
import os
import random
import joblib
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from .bpmn_parser import BPMNParser
from .log_analyzer import LogAnalyzer


class BranchPredictor:
    def __init__(self):
        self.probabilities = {}
        self.gateway_branches = {}
        self.gateway_connections = {}

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


def main():
    model_path = Path("../../models/branch_predictor.joblib")
    log_path = "../../Dataset/BPI Challenge 2017.xes"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run train.py first to train and save the model.")
        return

    print("Loading pre-trained model...")
    predictor = BranchPredictor.load(model_path)
    
    print(f"Loaded {len(predictor.probabilities)} decision points")
    print(f"Found {len(predictor.gateway_branches)} gateways")

    print("\nLoading log for testing...")
    analyzer = LogAnalyzer(log_path)
    
    activity_to_gateways = {}
    for gw_id, conn in predictor.gateway_connections.items():
        for act in conn['preceding']:
            if act not in activity_to_gateways:
                activity_to_gateways[act] = []
            activity_to_gateways[act].append({
                'gateway': gw_id,
                'branches': conn['branches']
            })

    correct = 0
    incorrect = 0
    tested = 0

    print("\nTesting predictions against actual traces...")
    
    for activities in analyzer.iter_traces():
        for i in range(len(activities) - 1):
            current = activities[i]
            actual_next = activities[i + 1]
            
            if current not in activity_to_gateways:
                continue
            
            for gw_info in activity_to_gateways[current]:
                if actual_next not in gw_info['branches']:
                    continue
                
                predicted = predictor.predict(gw_info['gateway'], current)
                
                if predicted == actual_next:
                    correct += 1
                else:
                    incorrect += 1
                tested += 1
                
                if tested <= 10:
                    status = "OK" if predicted == actual_next else "MISS"
                    print(f"  [{status}] After '{current[:25]}' -> actual: {actual_next[:20]}, pred: {predicted[:20]}")

    print(f"\nResults:")
    print(f"  Tested: {tested}")
    print(f"  Correct: {correct}")
    print(f"  Incorrect: {incorrect}")
    if tested > 0:
        print(f"  Accuracy: {correct/tested:.1%}")


if __name__ == "__main__":
    main()






