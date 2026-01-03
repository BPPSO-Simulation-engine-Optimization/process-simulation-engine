import random
import joblib
from pathlib import Path


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

    def get_probabilities(self, gateway_id, preceding_activity):
        key = (gateway_id, preceding_activity)
        return self.probabilities.get(key, {})


def simulate_case(predictor):
    """Simulate a single loan application case."""
    current_activity = "A_Create Application"
    trace = [current_activity]
    
    # Example decision points (use real gateway IDs from your model)
    decisions = [
        ("Gateway_abc123", "A_Create Application"),
        ("Gateway_def456", "W_Complete application"),
    ]
    
    print(f"Starting case with: {current_activity}")
    
    for gateway_id, preceding in decisions:
        probs = predictor.get_probabilities(gateway_id, preceding)
        if probs:
            print(f"\n  At gateway {gateway_id[:15]}... after '{preceding[:20]}'")
            print(f"  Options: {probs}")
        
        next_activity = predictor.predict(gateway_id, preceding)
        if next_activity:
            trace.append(next_activity)
            print(f"  -> Selected: {next_activity}")
    
    return trace


def main():
    model_path = Path(__file__).parent.parent.parent / "models" / "branch_predictor.joblib"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run train.py first to train and save the model.")
        return
    
    # Load once at simulation start
    print("Loading branch predictor model...")
    predictor = BranchPredictor.load(model_path)
    print(f"Ready! ({len(predictor.probabilities)} decision points loaded)\n")
    
    # Simulate multiple cases
    print("=" * 50)
    print("Simulating 3 cases...")
    print("=" * 50)
    
    for i in range(3):
        print(f"\n--- Case {i+1} ---")
        trace = simulate_case(predictor)
        print(f"  Final trace: {' -> '.join(trace)}")


if __name__ == "__main__":
    main()







