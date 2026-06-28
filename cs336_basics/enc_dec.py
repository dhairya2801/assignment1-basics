import regex as re
import pickle

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=[]):
        self.vocab = vocab # id -> token
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_enc = {}

        # reverse index for encoding token -> id
        for k, v in vocab.items():
            self.vocab_enc[v] = k
    
    def encode_iterable(self, iterable):
        pass
        
    def encode(self, text):
        '''
        1. pre-tokenize and represent each pre-token as a sequence of bytes
        2. apply the merges to pre-tokens (respect the order of merge creation) until no more can be applied
        3. represent as token IDs
        
        '''
        # split on special tokens
        SPECIAL_TOKEN = "<|endoftext|>"
        docs = re.split(re.escape(SPECIAL_TOKEN), text)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        ids = []

        for d in range(len(docs)):

            # pre-tokenize
            matches = re.finditer(PAT, docs[d])

            # represent as sequence of bytes
            pretokenized_text = []
            for match in matches:
                tup = ()
                word = match.group().encode('utf-8')

                for b in word:
                    tup += (bytes([b]), )
                                            
                pretokenized_text.append(tup)

            # apply merges to every pre-token
            for i in range(len(pretokenized_text)):
                for merge in self.merges:
                    a = merge[0]
                    b = merge[1]
                    word = pretokenized_text[i]
                    tup = ()
                    if a in word and b in word:
                        # perform merge
                        j = 0
                        while j < len(word):
                            if j+1 < len(word) and word[j] == a and word[j+1] == b:
                                merged = a+b
                                tup += (merged, )
                                j += 2
                            else:
                                tup += (word[j], )
                                j += 1
                        pretokenized_text[i] = tup
            
            if len(docs)>1 and d < len(docs)-1:
                pretokenized_text.append(("<|endoftext|>".encode("utf-8"), ))

            # convert to token ids
            for i in pretokenized_text:
                for j in i:
                    ids.append(self.vocab_enc[j])
                            
        return ids
    
    def decode(self, ids):
        out_text = b''
        for i in ids:
            out_text += self.vocab[i]
        
        out_text = out_text.decode('utf-8', errors='replace')

        return out_text

    
if __name__ == '__main__':
    with open("../tokenizer_tinystories_naive.pickle", "rb") as infile:
        merge = pickle.load(infile)
    
    tok = Tokenizer(merge['vocab'], merge['merges'])
    text = (
        'Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play '
    'near the pit. One day, Tom lost his red ball. He was very sad. Tom asked his friend, Sam, to help him search '
    'for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell '
    'into the pit." Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. '
    'They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball." '
    'They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get '
    'out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear '
    'them. They were sad and scared, and they never got out of the pit.'
    )

    text = '🙃'
    
    out = tok.encode(text)
    dec = tok.decode(out)