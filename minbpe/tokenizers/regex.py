## This file contains the train and infer code for a Regex Tokenizer.
##  We also handle special tokens

import regex as re
from .base import get_stats, merge_token, Tokenizer

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

class RegexTokenizer(Tokenizer):
    def __init__(self, special_tokens={}, model=None):
        super().__init__()
        self.register_special_tokens(special_tokens)
        
        if model == "gpt2":
            self.pattern = GPT2_PATTERN
        elif model == "gpt4" or model is None:
            self.pattern = GPT4_PATTERN
        else:
            raise NotImplementedError
        
        self.pattern_regex = re.compile(self.pattern)
        self.special = "(" + "|".join(re.escape(k) for k in self.special_tokens.keys()) + ")"
        self.vocab = self._build_vocab()
        self.byte_shuffle = {i: i for i in range(256)} 
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inv_special_tokens = {v:k for k, v in self.special_tokens.items()}

    def train(self, text, vocab_size, verbose=False, display_iter=None):
        assert vocab_size > len(self.vocab)
        num_merges = vocab_size - len(self.vocab)

        # split the text
        split_text = self.pattern_regex.findall(text)
        # encode
        tokens_list = [list(t.encode("utf-8")) for t in split_text]
        orig_size = sum([len(tokens) for tokens in tokens_list])
        if verbose:
            print(f"Start | Number of tokens: {orig_size}")

        for merge in range(num_merges):
            # get stats per tokens
            total_stats = {}
            for tokens in tokens_list:
                stats = get_stats(tokens)
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v


            max_freq_pair = max(total_stats, key=total_stats.get)

            # now merge this token and create new tokens 
            max_freq_pair_id = 256 + merge
            for i in range(len(tokens_list)):
                tokens_list[i] = merge_token(tokens_list[i], max_freq_pair, max_freq_pair_id)
            # update running stats
            self.merges[max_freq_pair] = max_freq_pair_id
            first_token, second_token = self.vocab[max_freq_pair[0]], self.vocab[max_freq_pair[1]] 
            self.vocab[max_freq_pair_id] = first_token + second_token

            if verbose:
                if display_iter is None:
                    display_iter = 1
                if (merge % display_iter == 0) or (merge == num_merges - 1):
                    print(f"Iteration {merge:4d} | Number of tokens: {sum([len(tokens) for tokens in tokens_list]):8d} | Merge: {first_token.decode('utf-8', errors='replace')} + {second_token.decode('utf-8', errors='replace')} --> {max_freq_pair_id}")

        if verbose:
            print(f"Compression : {(orig_size / sum([len(tokens) for tokens in tokens_list])):.2f} X")

    def encode_ordinary(self, text):
        split_text = self.pattern_regex.findall(text)

        tokens_list = [list(t.encode("utf-8")) for t in split_text]
        # now shuffle the byte tokens
        shuffled_tokens_list = [[self.byte_shuffle[b] for b in tokens] for tokens in tokens_list]
        encoded_tokens_list = []

        num_merges = 0
        for tokens in shuffled_tokens_list:
            while len(tokens) >= 2:
                # first get the stats of the bigrams
                pair_stats = get_stats(tokens)
                # now check if there is a pair which is merged as per our tokenizer
                merge_pair_candidate = min(pair_stats, key=lambda k: self.merges.get(k, float("inf"))) # it will check if we get a merge pair candidate, else returns the first element
                # check if there is actually a match
                if merge_pair_candidate not in self.merges:
                    break

                # now replace with the merges token
                tokens = merge_token(tokens, merge_pair_candidate, self.merges[merge_pair_candidate])
                num_merges += 1
            encoded_tokens_list.append(tokens)

        final_tokens = [item for sublist in encoded_tokens_list for item in sublist]

        return final_tokens, num_merges

    def encode(self, text, allowed_special="none", verbose=False):
        # first identify if special tokens should be handles or not
        # if none, encode ordinary :)
        num_merges = 0
        if allowed_special == "none":
            tokenized_text, num_merges = self.encode_ordinary(text)
            if verbose:
                print(f"Number of merges = {num_merges}")
            return tokenized_text
        elif allowed_special != "all":
            raise NotImplementedError
        
        # else first handle special token
        special_split_text = re.split(self.special, text)

        # next tokenize these individual non special ones and add them to split_text list
        final_tokens = []
        for t in special_split_text:
            if t in self.special_tokens:
                final_tokens.append(self.special_tokens[t])
            else:
                tokenized_text, num_merges_tmp = self.encode_ordinary(t)
                final_tokens.extend(tokenized_text)
                num_merges += num_merges_tmp


        if verbose:
            print(f"Number of merges = {num_merges}")
        return final_tokens
        

    def decode(self, ids):
        enc_text_split = []
        for id in ids:
            if id in self.vocab:
                enc_text_split.append(self.vocab[id])
            elif id in self.inv_special_tokens:
                enc_text_split.append(self.inv_special_tokens[id].encode("utf-8"))
        enc_text = b"".join(t for t in enc_text_split)
        print(enc_text)
        text = enc_text.decode("utf-8", errors="replace")
        return text

        