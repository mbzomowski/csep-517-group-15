"""
Prepare text datasets for character-level language modeling.
Scans all .txt files in a given directory, maps characters to integers,
and saves train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder mappings.
"""
import os
import pickle
import numpy as np
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Prepare text files for character-level language modeling")
    parser.add_argument('--input_dir', type=str, default='data/flores', 
                        help='Directory containing .txt files to process')
    parser.add_argument('--output_dir', type=str, default='work',
                        help='Directory to save output files (meta.pkl, train.bin, val.bin)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .txt files in the input directory
    input_files = glob.glob(os.path.join(args.input_dir, '*.txt'))
    
    if not input_files:
        print(f"No .txt files found in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} text files to process")
    
    stoi = dict()
    itos = dict()
    start = 0
    seen_chars = set()
    
    train_data = ""
    val_data = ""
    vocab_size = 0
    
    # Process each file
    for file in input_files:
        print(f"Processing {os.path.basename(file)}")
        with open(file, 'r', encoding='utf-8') as f:
            try:
                data = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Skipping {file} due to encoding issues")
                continue
                
        print(f"  - Length of file in characters: {len(data):,}")
    
        # get all the unique characters that occur in this text
        data_set = set(data)
        data_set = data_set.difference(seen_chars)
        unique_data_size = len(data_set)
    
        chars = sorted(list(data_set))
        vocab_size += len(chars)
        print(f"  - New unique characters: {len(chars)}")
        print(f"  - Total vocab size so far: {vocab_size:,}")
    
        # create a mapping from characters to integers
        stoi.update({ch: i for i, ch in enumerate(chars, start=start)})
        itos.update({i: ch for i, ch in enumerate(chars, start=start)})
    
        start += unique_data_size
        seen_chars = seen_chars.union(data_set)
        
        # create the train and test splits
        n = len(data)
        train_data += '\n' + data[:int(n*0.9)]
        val_data += '\n' + data[int(n*0.9):]
    
    # Skip first newline if present
    if train_data.startswith('\n'):
        train_data = train_data[1:]
    if val_data.startswith('\n'):
        val_data = val_data[1:]
    
    def encode(s):
        return [stoi.get(c, 0) for c in s]  # encoder: take a string, output a list of integers
    
    def decode(l):
        return ''.join([itos.get(i, '') for i in l])  # decoder: take a list of integers, output a string
    
    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"Train set has {len(train_ids):,} tokens")
    print(f"Validation set has {len(val_ids):,} tokens")
    
    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_bin_path = os.path.join(args.output_dir, 'train.bin')
    val_bin_path = os.path.join(args.output_dir, 'val.bin')
    meta_path = os.path.join(args.output_dir, 'meta.pkl')
    
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)
    
    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"\nSaved output files to {args.output_dir}:")
    print(f"  - {os.path.basename(meta_path)}")
    print(f"  - {os.path.basename(train_bin_path)}")
    print(f"  - {os.path.basename(val_bin_path)}")

if __name__ == "__main__":
    main()
