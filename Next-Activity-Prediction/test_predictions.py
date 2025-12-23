import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from bpmn_parser import BPMNParser
from log_analyzer import LogAnalyzer
import random


class TestPredictor:
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
    bpmn_path = "../process_model/LoanApplicationProcess.bpmn"
    log_path = "../Dataset/BPI Challenge 2017.xes"

    print("Training predictor...")
    predictor = TestPredictor()
    predictor.fit(bpmn_path, log_path)
    
    print(f"Learned {len(predictor.probabilities)} decision points")
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

