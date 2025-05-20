"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np


# download the tiny shakespeare dataset
eng_input_file_path = os.path.join(os.path.dirname(__file__), 'eng_Latn.txt')
fra_input_file_path = os.path.join(os.path.dirname(__file__), 'fra_Latn.txt')
jpn_input_file_path = os.path.join(os.path.dirname(__file__), 'jpn_Jpan.txt')

files = []

files.append(eng_input_file_path)
files.append(fra_input_file_path)
files.append(jpn_input_file_path)

stoi = dict()
itos = dict()
start = 0
seen_chars = set()

train_data = ""
val_data = ""
vocab_size = 0


for file in files:
    with open(file, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    data_set = set(data)
    data_set = data_set.difference(seen_chars)
    unique_data_size = len(data_set)

    chars = sorted(list(data_set))
    vocab_size += len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi.update({ch: i for i, ch in enumerate(chars, start=start)})
    itos.update({i: ch for i, ch in enumerate(chars, start=start)})

    start += unique_data_size
    seen_chars = seen_chars.union(data_set)
    # create the train and test splits
    n = len(data)
    train_data += '\n' + data[:int(n*0.9)]
    val_data += '\n' + data[int(n*0.9):]


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
