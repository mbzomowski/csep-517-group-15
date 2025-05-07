#!/usr/bin/env python
import os
import sys
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
import subprocess
import pickle
import torch

# script_dir  = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from model import GPT, GPTConfig



class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
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

    def run_train(self, data, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        cmd = [
            "python", "train.py",
            "config/train_flores_eng_char.py",
            f"--out_dir={work_dir}",
            "--device=cpu",
            "--compile=False",
            "--batch_size=12",
            "--block_size=128",
            "--max_iters=200",
            "--eval_interval=200",
            "--dataset=flores"
        ]
        print("Running training:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def run_pred(self, data):
        import os, sys, pickle, torch

        here       = os.path.dirname(__file__)                    
        project    = os.path.abspath(os.path.join(here, os.pardir))  
        if project not in sys.path:
            sys.path.insert(0, project)
        from model import GPT, GPTConfig

        # ─── hard-coded paths ───────────────────────────────────────────────
        meta_path  = "data/flores/meta.pkl"
        ckpt_path  = "out-flores-eng/ckpt.pt"

        # 1) load vocab mappings
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi = meta["stoi"]   # char → idx
        itos = meta["itos"]   # idx  → char

        # 2) load checkpoint and rebuild model
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

    

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        inst = MyModel()
        inst.work_dir = work_dir    # ← we save it on the instance
        return inst


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train','test'))
    parser.add_argument('--work_dir',    default='work')
    parser.add_argument('--test_data',   default='example/input.txt')
    parser.add_argument('--test_output', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        os.makedirs(args.work_dir, exist_ok=True)
        model = MyModel()
        train_data = model.load_training_data()
        model.run_train(train_data, args.work_dir)
    else:
        model = MyModel.load(args.work_dir)
        test_data = model.load_test_data(args.test_data)
        preds     = model.run_pred(test_data)
        assert len(preds)==len(test_data)
        model.write_pred(preds, args.test_output)
