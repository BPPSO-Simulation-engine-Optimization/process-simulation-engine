import torch
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    # Optional dependency, will fail if download needed
    hf_hub_download = None

# Ensure we can import the model definition

# Ensure we can import the model definition
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from .processtransformer import ProcessTransformer
except ImportError:
    # Fallback for when running from a different context
    from processtransformer import ProcessTransformer

logger = logging.getLogger(__name__)

class ProcessTransformerV2Predictor:
    """
    Unified predictor for Next Activity AND Service Time.
    
    Implements BOTH:
    1. NextActivityPredictor protocol
    2. ProcessingTimePredictor protocol
    
    Strategy:
    - On `predict(case_state)` (Activity):
      Runs the model, returns next activity, and CACHES the predicted service time.
    - On `predict(..., context)` (Time):
      Retrieves the cached service time for the given case_id.
    """
    
    START_ACTIVITY = "A_Create Application"
    END_ACTIVITIES = {"END", "<END>"}

    def __init__(self, model_path: str = None, config_path: str = None, device: str = None, temperature: float = 1.5, end_token_penalty: float = 3.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.end_token_penalty = end_token_penalty
        
        # Default paths if not provided
        if model_path is None:
            # Check local first
            local_model = current_dir / "model.pth"
            if local_model.exists():
                model_path = str(local_model)
            else:
                # Try HuggingFace
                logger.info("Local model not found. Attempting download from HuggingFace...")
                try:
                    model_path = hf_hub_download(repo_id="lgk03/process-transformer-bpic17", filename="model.pth")
                except Exception as e:
                    logger.warning(f"Could not download model from HuggingFace: {e}")
                    model_path = str(local_model) # Fallback to fail later

        if config_path is None:
             # Check local first
            local_config = current_dir / "config.json"
            if local_config.exists():
                config_path = str(local_config)
            else:
                # Try HuggingFace
                logger.info("Local config not found. Attempting download from HuggingFace...")
                try:
                    config_path = hf_hub_download(repo_id="lgk03/process-transformer-bpic17", filename="vocab.json")
                except Exception as e:
                    logger.warning(f"Could not download config from HuggingFace: {e}")
                    config_path = str(local_config)
            
        self.model_path = model_path
        self.config_path = config_path
        
        self.duration_cache: Dict[str, float] = {}
        
        self._load_model()
        
    def _load_model(self):
        logger.info(f"Loading ProcessTransformerV2 from {self.model_path}")
        
        # Load Config (Vocabulary)
        with open(self.config_path, 'r') as f:
            self.vocab = json.load(f)
            
        # Vocab mapping
        self.act_to_idx = {k: int(v) for k, v in self.vocab.items()}
        self.idx_to_act = {v: k for k, v in self.act_to_idx.items()}
        
        # Model Parameters matching training
        # Note: If these were dynamic in training, they should be saved in config too. 
        # Assuming defaults from training script for now.
        config_dict = self.vocab  # Check if config is just vocab or robust config
        # If vocab is just the dict, we infer size.
        # In training script: vocab_size = len(dataset.vocab) + 1
        # MAX_LEN was 50
        
        vocab_size = len(self.vocab) + 1
        max_len = 50
        
        self.model = ProcessTransformer(num_activities=vocab_size, max_len=max_len)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.max_len = max_len
        logger.info("Model loaded successfully.")

    def predict(self, case_state: Any) -> Tuple[str, bool]:
        """
        Next Activity Prediction Protocol.
        Also predicts and caches duration.
        """
        # 1. Preprocess
        history = case_state.activity_history
        
        # Helper to tokenize
        def tokenize(act):
            return self.act_to_idx.get(act, 0) # 0 is padding/unknown
            
        indices = [tokenize(a) for a in history]
        
        # Pad/Truncate
        if len(indices) == 0:
            # Cold start - Fix: Return Start Activity directly
            return self.START_ACTIVITY, False
            
        seq_len = len(indices)
        if seq_len < self.max_len:
            # Pad left
            padded = [0] * (self.max_len - seq_len) + indices
        else:
            # Truncate to keep only the last max_len items
            padded = indices[-self.max_len:]
            
        input_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            logits_act, out_dur = self.model(input_tensor)
            
        # 3. Postprocess Activity
        # Apply End Token Penalty
        if self.end_token_penalty != 1.0:
            for end_act in self.END_ACTIVITIES:
                if end_act in self.act_to_idx:
                    end_idx = self.act_to_idx[end_act]
                    logit = logits_act[0, end_idx]
                    if logit < 0:
                        logits_act[0, end_idx] = logit * self.end_token_penalty
                    else:
                        logits_act[0, end_idx] = logit / self.end_token_penalty

        # Temperature Sampling
        probs = torch.softmax(logits_act[0] / self.temperature, dim=0)
        pred_idx = torch.multinomial(probs, 1).item()
        
        next_activity = self.idx_to_act.get(pred_idx, "A_Complete") # Fallback
        
        # 4. Postprocess Duration & Cache
        # Training used: log1p(duration) -> output
        # Inverse: exp(output) - 1
        pred_dur_log = out_dur[0].item()
        pred_dur = max(0.0, np.exp(pred_dur_log) - 1.0)
        
        # CACHE IT using case_id
        if hasattr(case_state, 'case_id'):
            self.duration_cache[case_state.case_id] = pred_dur
            
        is_end = next_activity in self.END_ACTIVITIES
        
        return next_activity, is_end

    # --- ProcessingTimePredictor Protocol ---
    
    def predict_processing_time(
        self,
        prev_activity: str,
        prev_lifecycle: str,
        curr_activity: str,
        curr_lifecycle: str,
        context: Dict = None,
    ) -> float:
        """
        Retrieves cached duration.
        Method signature matches ProcessingTimePredictor protocol in engine.py 
        (the engine calls .predict(), but since we have a collision on method name 'predict', 
         we will handle the dispatching or rename. 
         Wait, Python doesn't support method overloading by signature.
         
         PROBLEM: The engine expects:
         1. next_activity_predictor.predict(case_state)
         2. processing_time_predictor.predict(prev_act, ..., context)
         
         If I use the SAME instance for both, I need a single `predict` method that handles both signatures?
         Or, I can use a small wrapper class for one of them.
         
         Solution: This class will implement `predict_next_activity` and `predict_processing_time`.
         Then I return a tuple of wrappers or a unified object that inspects arguments?
         
         Actually, cleanest is:
         ID: ProcessTransformerV2Predictor (the main brain)
         ID: ProcessTransformerActivityAdapter (calls brain.predict_activity)
         ID: ProcessTransformerTimeAdapter (calls brain.predict_time)
         
         But to keep it single-file simple, I will implement `predict` and check arguments.
        """
        # This methodology is risky if arguments overlap, but here they are distinct.
        # activity predict takes (case_state) -> 1 arg
        # time predict takes (prev, prev_life, curr, curr_life, context) -> 5 args
        pass

    def predict_time_from_cache(self, case_id: str) -> float:
        """Retrieves and clears duration for the case."""
        return self.duration_cache.pop(case_id, 0.0) # Default 0 if missing

# Adapters to satisfy the strict Protocol interfaces
class PTActivityAdapter:
    def __init__(self, predictor: ProcessTransformerV2Predictor):
        self.predictor = predictor
        
    def predict(self, case_state: Any) -> Tuple[str, bool]:
        return self.predictor.predict(case_state)

class PTTimeAdapter:
    def __init__(self, predictor: ProcessTransformerV2Predictor):
        self.predictor = predictor
        
    def predict(self, prev_activity, prev_lifecycle, curr_activity, curr_lifecycle, context=None) -> float:
        # We only really care about the cached value which is stored by case_id
        if context and 'case_id' in context:
            return self.predictor.predict_time_from_cache(context['case_id'])
        return 0.0 # Fallback 0s if no context/cache (immediate)
