from collections import defaultdict
'''
BPE Training steps:
1. vocab init
2. pre-tokenization
3. compute bpe merges
'''

text = 'low low low low low lower lower widest widest widest newest newest newest newest newest newest'

vocab = [bytes([i]) for i in range(256)]

# special token
vocab.append('<|endoftext|>')

text_list = text.split(' ')

pairs = ()
pre_tokens = defaultdict(int)

for i in text_list:
    tup = ()
    for j in i:
        b = j.encode('utf-8')
        tup += (b,)
    
    pre_tokens[tup] += 1

def create_pairs(pre_tokens):
    pairs = defaultdict(int)

    for k, v in pre_tokens.items():
        for j in range(len(k)-1):
            pairs[(k[j], k[j+1])] += v

    return pairs


def merge(pairs:defaultdict):
    '''
    if tie, return the lexicographically greater one.
    time: O(n)
    '''
    pair = ()
    maxF = 0
    for k, v in pairs.items():
        if v > maxF:
            pair = k
            maxF = v
        elif v == maxF:
            if k > pair:
                pair = k
    
    return pair

def resolve_pairs(pre_tokens:defaultdict, pair:tuple):
    '''
    1. resolve pairs in pre_token dict
    2. remake pairs
    '''
    new_pre_tokens = defaultdict(int)
    for k, v in pre_tokens.items():
        newK = ()
        j = 0
        while j < len(k):
            if j < len(k)-1 and pair == (k[j], k[j+1]):
                newMerge = pair[0]+pair[1]
                newK += (newMerge,)
                j += 2
            else:
                newK += (k[j],)
                j += 1
        new_pre_tokens[newK] = v
    
    pre_tokens = new_pre_tokens
    
    return pre_tokens

# bpe training loop
for i in range(6):
    pairs = create_pairs(pre_tokens)
    pair = merge(pairs)
    pre_tokens = resolve_pairs(pre_tokens, pair)

# print(pre_tokens)