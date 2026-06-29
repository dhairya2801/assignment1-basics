import os
import resource
import time
import pickle
from collections import defaultdict

from pretokenization_example import counts

'''
input: pre-token counts: dict[tuple[bytes], count], max vocab size: int, special tokens: list[str]
output: vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
'''

def create_pairs(pretoken_counts):
    pairs = defaultdict(int)
    reverse_index = defaultdict(list)

    for tup, count in pretoken_counts.items():
        for i in range(len(tup)-1):
            t = (tup[i], tup[i+1])
            pairs[t] += count
            
            # reverse_index[t].append(tup)
    
    return pairs, reverse_index

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
    '''
    1. reverse index (dict): key: pair, value: list of sequences containing pair
    2. if pair in rev_ind, iterate through all sequences and update global_counts, the pairs dict,
    the reverse index itself
    '''
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
    # todo: handling a list of special tokens

    # initiate base vocab
    vocab = [bytes([i]) for i in range(256)]
    spec_tok = "<|endoftext|>".encode("utf-8") # only for TinyStories
    vocab.append(spec_tok)

    merges = []

    num_processes = os.cpu_count() - 1
    
    # pretokenization
    global_counts, pretok_time = counts(path, num_processes)

    start = time.perf_counter()

    while len(vocab) < max_vocab_size:
        # create pairs with frequencies
        pairs, reverse_index = create_pairs(global_counts)
        
        pair = pair_to_merge(pairs)
        if len(pair) < 2:
            break

        merges.append(pair)
        
        merged = pair[0]+pair[1]
        vocab.append(merged)

        global_counts = resolve_pairs(global_counts, pair)
    
    end = time.perf_counter()

    training_time = end - start

    vocab_dict = {}
    # print(reverse_index)

    for i in range(len(vocab)):
        vocab_dict[i] = vocab[i]

    # find longest token in the vocab
    longest = ''
    l = 0
    for i in vocab:
        if len(i) > l:
            l = len(i)
            longest = i
    
    # find peak rss memory
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss = peak_rss / (1024 ** 2)
    
    # serialize data for future use
    serialize_data = {}
    serialize_data["vocab"] = vocab_dict
    serialize_data["merges"] = merges
    serialize_data["longest_token"] = longest
    serialize_data["training_time"] = f'{training_time:.2f} s'
    serialize_data["pretokenization_time"] = f'{pretok_time:.2f} s'
    serialize_data["peak_rss"] = f'{peak_rss:.2f} GiB'
    
    with open("tokenizer_owt_naive.pickle", "wb") as out_vocab:
        pickle.dump(serialize_data, out_vocab)
    
    return vocab_dict, merges
        
if __name__ == '__main__':
    path = "/home/dhairya2801/Dhairya/cs336/assignment1-basics/data/owt_train.txt"
    merge(path, 32000, ['<|endoftext|>'])