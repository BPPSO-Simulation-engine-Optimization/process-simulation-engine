"""
Train and save the branch prediction model.
Run this once before simulation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .bpmn_parser import BPMNParser
from .log_analyzer import LogAnalyzer
import joblib
from pathlib import Path


def train_and_save(bpmn_path, log_path, output_path):
    print("Parsing BPMN model...")
    parser = BPMNParser(bpmn_path)
    gateway_connections = parser.get_gateway_connections()
    gateway_branches = {
        gw_id: conn['branches']
        for gw_id, conn in gateway_connections.items()
    }

    print("Analyzing event log...")
    analyzer = LogAnalyzer(log_path)
    counts = analyzer.count_transitions(gateway_connections)
    probabilities = analyzer.calculate_probabilities(counts)

    print(f"Learned {len(probabilities)} decision points")
    print(f"Found {len(gateway_branches)} gateways")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'probabilities': probabilities,
        'gateway_branches': gateway_branches,
        'gateway_connections': gateway_connections
    }
    joblib.dump(data, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    bpmn_path = "../../process_model/LoanApplicationProcess.bpmn"
    log_path = "../../Dataset/BPI Challenge 2017.xes"
    output_path = "../../models/branch_predictor.joblib"

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_and_save(bpmn_path, log_path, output_path)








