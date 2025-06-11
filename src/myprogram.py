#!/usr/bin/env python
import os
import sys
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess
import pickle
import torch


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    
    def __init__(self):
        self.model = None
        self.stoi = None
        self.itos = None
        self.device = None
        self.work_dir = None

    @classmethod
    def load_training_data(cls):
        return []

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data
    
    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def _load_model_once(self):
        """Load model and metadata only once for efficiency"""
        if self.model is not None:
            return  # Already loaded
            
        # Add project root to path to ensure imports work
        here = os.path.dirname(__file__)                    
        project = os.path.abspath(os.path.join(here, os.pardir))  
        if project not in sys.path:
            sys.path.insert(0, project)
        from model import GPT, GPTConfig

        # Find meta.pkl file
        possible_meta_paths = [
            "/job/meta.pkl",               # Docker root directory
            "work/meta.pkl"                # work directory
        ]
        
        meta_path = None
        for path in possible_meta_paths:
            if os.path.exists(path):
                meta_path = path
                print(f"Found meta.pkl at: {meta_path}")
                break
                
        if meta_path is None:
            raise FileNotFoundError("Could not find meta.pkl in any of the expected locations")
            
        # Find checkpoint file
        possible_ckpt_paths = [
            "/job/ckpt.pt",                 # Docker root
            "work/ckpt.pt"                  # work directory
        ]
        
        ckpt_path = None
        for path in possible_ckpt_paths:
            if os.path.exists(path):
                ckpt_path = path
                print(f"Found ckpt.pt at: {ckpt_path}")
                break
                
        if ckpt_path is None:
            raise FileNotFoundError("Could not find ckpt.pt in any of the expected locations")

        # Load vocab mappings
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.stoi = meta["stoi"]   # char → idx
        self.itos = meta["itos"]   # idx  → char
        
        # Setup device for fast inference
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA for inference")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using MPS for inference")
            # Set MPS fallback for unsupported operations
            torch.backends.mps.allow_fallback = True
        else:
            self.device = torch.device('cpu')
            print("Using CPU for inference")

        # Load checkpoint and rebuild model
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = GPT(GPTConfig(**ckpt["model_args"]))
        
        # Fix the state dict keys by removing '_orig_mod' prefix
        state_dict = ckpt["model"]
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Compile model for faster inference (if supported and stable)
        # Skip compilation on MPS as it's still experimental
        if self.device.type != 'mps':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled for faster inference")
            except:
                print("Model compilation not available, using uncompiled model")
        else:
            print("Skipping model compilation on MPS (experimental), using uncompiled model")

    def run_pred(self, data): 
        """Optimized prediction with batching and device acceleration"""
        self._load_model_once()
        
        # Batch process for efficiency
        # Use smaller batch size for MPS to avoid memory issues
        batch_size = 16 if self.device.type == 'mps' else 32
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                batch_preds = self._predict_batch(batch_data)
                all_preds.extend(batch_preds)
                
        return all_preds
    
    def _predict_batch(self, batch_data):
        """Process a batch of inputs efficiently"""
        batch_preds = []
        
        # Find max length for padding
        max_len = max(len(line) for line in batch_data)
        
        # Encode all inputs in the batch
        batch_encoded = []
        for line in batch_data:
            encoded = [self.stoi.get(c, 0) for c in line]
            # Pad to max length
            padded = encoded + [0] * (max_len - len(encoded))
            batch_encoded.append(padded)
        
        # Convert to tensor and move to device
        x = torch.tensor(batch_encoded, dtype=torch.long, device=self.device)
        
        # Get predictions for the batch
        logits, _ = self.model(x)  # Shape: (batch_size, seq_len, vocab_size)
        
        # Get logits for the last position of each sequence
        last_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        # Get top 3 most likely next characters for each input
        probs = torch.nn.functional.softmax(last_logits, dim=-1)
        top_k = 3
        v, ix = torch.topk(probs, k=top_k, dim=-1)  # Shape: (batch_size, top_k)
        
        # Convert to characters
        for i, indices in enumerate(ix):
            next_chars = [self.itos[idx.item()] for idx in indices]
            pred = ''.join(next_chars)
            batch_preds.append(pred)
            
        return batch_preds

    @classmethod
    def load(cls, work_dir):
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        inst = MyModel()
        inst.work_dir = work_dir    # ← we save it on the instance
        return inst


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('test'))
    parser.add_argument('--work_dir', default='work')
    parser.add_argument('--test_data', default='example/input.txt')
    parser.add_argument('--test_output', default='output/pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'test':
        model = MyModel.load(args.work_dir)
        test_data = model.load_test_data(args.test_data)
        preds = model.run_pred(test_data)
        assert len(preds) == len(test_data)
        model.write_pred(preds, args.test_output)
    else:
        print(f"Mode '{args.mode}' is not implemented. Use 'test' mode for prediction.")
