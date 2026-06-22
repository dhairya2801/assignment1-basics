import os
from collections import defaultdict

from .pretokenization_example import counts

'''
input: pre-token counts: dict[tuple[bytes], count], max vocab size: int, special tokens: list[str]
output: vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
'''

def create_pairs(pretoken_counts):
    pairs = defaultdict(int)

    for tup, count in pretoken_counts.items():
        for i in range(len(tup)-1):
            pairs[(tup[i], tup[i+1])] += count
    
    return pairs

def pair_to_merge(pairs):
    '''
    1. count max freq
    2. return lexicographically greater pair
    '''
    maxF = 0
    pair = ()
    for tup, count in pairs.items():
        if count > maxF:
            pair = tup
            maxF = count
        elif count == maxF:
            pair = max(pair, tup)
    
    return pair

def resolve_pairs(global_counts, pair):
    new_global_counts = defaultdict(int)

    # naive implementation, iterating through all pairs to find merges
    for tup, count in global_counts.items():
        newTup = ()
        i = 0
        while i < len(tup):
            if i+1 < len(tup) and pair == (tup[i], tup[i+1]):
                merged = tup[i]+tup[i+1]
                newTup += (merged, )
                i += 2
            else:
                newTup += (tup[i], )
                i += 1
        
        new_global_counts[newTup] += count
    
    global_counts = new_global_counts
    
    return global_counts


def merge(path: str, max_vocab_size: int, special_tokens):
    # initiate base vocab
    vocab = [bytes([i]) for i in range(256)]
    spec_tok = "<|endoftext|>".encode("utf-8") # only for TinyStories
    vocab.append(spec_tok)

    merges = []

    num_processes = os.cpu_count() - 1
    global_counts = counts(path, num_processes)

    while len(vocab) < max_vocab_size:
        # create pairs with frequencies
        pairs = create_pairs(global_counts)
        
        pair = pair_to_merge(pairs)
        merges.append(pair)
        
        merged = pair[0]+pair[1] 
        vocab.append(merged)

        global_counts = resolve_pairs(global_counts, pair)
    
    vocab_dict = {}
    for i in range(len(vocab)):
        vocab_dict[i] = vocab[i]
    
    return vocab_dict, merges
        
if __name__ == '__main__':
    path = "/home/dhairya2801/Dhairya/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    merge(path, 10000, ['<|endoftext|>'])