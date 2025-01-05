## Implements gpt4 tokenizer read in from a file

from .regex import RegexTokenizer, GPT4_PATTERN
import tiktoken

import tiktoken

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self, special_tokens={
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}):
        super().__init__(special_tokens, model="gpt4")
        self.tokenizer_path = "cl100k_base"
        self.enc = tiktoken.get_encoding(self.tokenizer_path)
        self._create_vocab_and_merges()
        self.byte_shuffle = {i: self.enc._mergeable_ranks[bytes([i])] for i in range(256)} # does map actual byte 0 to the one in this dict

    def _get_split_tokens(self, mergeable_ranks, token, max_rank):
        # Idea: Get the most optimal split of the token into exactly two pre-existing tokens
        parts = [bytes([t]) for t in token]
        while True:
            min_id, min_rank = None, None # min_id -> the point to split the token, min_rank -> to find the most optimal split
            for i, pair in enumerate(zip(parts, parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])

                if (rank is not None) and (min_rank is None or rank < min_rank):
                    min_id = i
                    min_rank = rank

            if (min_rank is None) or (min_rank >= max_rank): # no split was obtained
                break
            # else update the tokens
            assert min_id is not None
            parts = parts[:min_id] + [parts[min_id] + parts[min_id + 1]] + parts[min_id + 2:]

        return parts

    def _create_vocab_and_merges(self):
        self.vocab = {v:k for k, v in self.enc._mergeable_ranks.items()}
        self.vocab_size = len(self.vocab)
        
        self.merges = {}
        for token, idx in self.enc._mergeable_ranks.items():
            if len(token) < 2:
                continue
            parts = self._get_split_tokens(self.enc._mergeable_ranks, token, idx)
            assert len(parts) == 2
            self.merges[(self.enc._mergeable_ranks[parts[0]], self.enc._mergeable_ranks[parts[1]])] = idx

    def train(self, text, vocab_size, verbose=False, verbose_iters=None):
        raise NotImplementedError