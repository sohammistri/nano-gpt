## This file contains the train and infer code for a Basic Tokenizer.
##  It implements a simple BPE tokenizer but does not handle the following:
##  - Regex match
##  - Special Tokens

from .base import get_stats, merge_token, Tokenizer

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False, display_iter=None):
        # check that vocab size is greater than the beginning (256)
        assert vocab_size > len(self.vocab)
        num_merges = vocab_size - len(self.vocab)
        
        # encode the text
        tokens = list(text.encode("utf-8"))
        orig_size = len(tokens)
        if verbose:
            print(f"Start | Number of tokens: {len(tokens)}")

        # start the merges
        for i in range(num_merges):
            # get the most frequent bigram
            stats = get_stats(tokens)
            max_freq_pair = max(stats, key=stats.get)

            # now merge this token and create new tokens 
            max_freq_pair_id = len(self.vocab)
            tokens = merge_token(tokens, max_freq_pair, max_freq_pair_id)

            # update running stats
            self.merges[max_freq_pair] = max_freq_pair_id
            first_token, second_token = self.vocab[max_freq_pair[0]], self.vocab[max_freq_pair[1]] 
            self.vocab[max_freq_pair_id] = first_token + second_token
            if verbose:
                if display_iter is None:
                    display_iter = 1
                if (i % display_iter == 0) or (i == num_merges - 1):
                    print(f"Iteration {i:4d} | Number of tokens: {len(tokens):8d} | Merge: {first_token.decode('utf-8', errors='replace')} + {second_token.decode('utf-8', errors='replace')} --> {max_freq_pair_id}")

        if verbose:
            print(f"Compression : {(orig_size / len(tokens)):.2f} X")

    def encode(self, text, verbose=False):
        # encode a given piece of text
        tokens = list(text.encode("utf-8"))
        num_merges = 0
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            merge_pair_candidate = min(stats, key=lambda k:self.merges.get(k, float("inf")))
            if merge_pair_candidate not in self.merges:
                break

            tokens = merge_token(tokens, merge_pair_candidate, self.merges[merge_pair_candidate])
            num_merges += 1

        if verbose:
            print(f"Number of merges = {num_merges}")

        return tokens
    
    def decode(self, tokens):
        enc_text = b"".join(self.vocab[token] for token in tokens)
        text = enc_text.decode("utf-8", errors="replace")
        return text 