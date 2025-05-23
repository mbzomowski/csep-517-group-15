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

    def run_pred(self, data): 
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
        stoi = meta["stoi"]   # char → idx
        itos = meta["itos"]   # idx  → char

        # Load checkpoint and rebuild model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = GPT(GPTConfig(**ckpt["model_args"]))
        
        # Fix the state dict keys by removing '_orig_mod' prefix
        state_dict = ckpt["model"]
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        model.load_state_dict(state_dict)
        model.eval()

        # Run prediction
        preds = []
        with torch.no_grad():
            for line in data:
                # encode the input string
                x = torch.tensor([stoi.get(c, 0) for c in line], dtype=torch.long)[None, :]
                
                # get logits for next character only
                logits, _ = model(x)  # model returns (logits, loss)
                logits = logits[:, -1, :]  # shape: (1, vocab_size)
                
                # get top 3 most likely next characters
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_k = 3
                v, ix = torch.topk(probs, k=top_k)
                
                # convert to characters and join
                next_chars = [itos[i] for i in ix[0].tolist()]
                pred = ''.join(next_chars)
                preds.append(pred)

        return preds

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
