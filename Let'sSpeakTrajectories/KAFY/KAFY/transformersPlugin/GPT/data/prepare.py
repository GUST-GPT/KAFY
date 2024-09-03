"""
Prepare the trajectory dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
    data_tokens = data.split()
# print(f"length of dataset in characters: {len(data):,}")
print(f"length of dataset in tokens: {len(data_tokens):,}")

# get all the unique characters that occur in this text
# chars = sorted(list(set(data)))
# vocab_size = len(chars)
# print("all the unique characters:", ''.join(chars))
# print(f"vocab size: {vocab_size:,}")

# get all the unique tokens that occur in this text
space_token = ' '
newline_token = '\n'
# tokens = sorted(set({space_token,newline_token}))
tokens = sorted((set(data_tokens)))
tokens.insert(0,newline_token)
tokens.insert(1,space_token)
vocab_as_tokens_size = len(tokens)

# print(tokens)
# print("all the unique tokens:", ''.join(tokens))
print(f"vocab in terms of tokens size (# Of Unique Tokens): {vocab_as_tokens_size:,}")

# create a mapping from tokens to integers
stoi = { ch:i for i,ch in enumerate(tokens) }
itos = { i:ch for i,ch in enumerate(tokens) }


# Add the space and new_line to tokens
# stoi[space_token] = len(tokens) - 1
# stoi[newline_token] = len(tokens) - 1

# print(stoi[' '])
# print(stoi['\n'])
# # Add the new element to stoi with the index equal to the length of tokens - 1

def encode(s):
    return [stoi[c] for c in s] # encoder: take a trajectory "Statement", output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a trajectory "Statement"

# create the train and test splits
n = len(data_tokens)
train_data = data_tokens[:int(n*0.9)]
val_data = data_tokens[int(n*0.9):]
# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_as_tokens_size,
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


