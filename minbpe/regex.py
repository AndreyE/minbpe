"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

from collections import defaultdict
import regex as re
import time

from .base import Tokenizer, get_stats, merge


SIZEOF_CHAR = 256


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _collect_chunks(lines, pattern):
        vocab = {0: b'\xef\xbf\xbd'} # '�'
        char_ids = defaultdict(int)

        # split the text up into text chunks
        def chars_gen(lines):
            count = 0
            mib = 0
            start_time = time.perf_counter()
            try:
                for line in lines:
                    for char in line:
                        b_char = char.encode('utf-8')
                        if char not in char_ids:
                            idx = len(vocab) + 1
                            vocab[idx] = b_char # TODO: get rid of extra encoding/decoding
                            char_ids[char] = idx
                        count += len(b_char)
                        if count // 1024**2 >= mib:
                            end_time = time.perf_counter()
                            print(f'Reading {count // 1024**2} MiB in {end_time - start_time} sec.')
                            start_time = end_time
                            mib += 1
                        yield char
            except UnicodeDecodeError as err:
                print(f'Skipping {line} due to {err}')

        def id_chunks_gen(lines):
            buffer = []
            for char in chars_gen(lines):
                try:
                    buffer += char
                    text_chunks = pattern.findall(''.join(buffer))
                    if len(text_chunks) > 1:
                        del buffer[:len(text_chunks[0])]
                        yield tuple(char_ids[ch] for ch in text_chunks[0])
                except UnicodeDecodeError as err:
                    print(err)

            if buffer:
                for chunk in pattern.findall(''.join(buffer)):
                    yield chunk

        chunks = {}
        for chunk in id_chunks_gen(lines):
            chunks[chunk] = chunks.get(chunk, 0) + 1
        return char_ids, vocab, chunks


def _count_pairs(chunks):
    pairs = {}
    for chunk, count in chunks.items():
        for pair in zip(chunk[:-1], chunk[1:]):
            pair = tuple(pair)
            pairs[pair] = pairs.get(pair, 0) + count
    return pairs


def _replace_pair(pair, chunks, idx):
    def replacer(chunk):
        i = 0
        while i < len(chunk) - 1:
            if chunk[i:i+2] == pair:
                yield idx
                i += 2
            else:
                yield chunk[i]
                i += 1

        if i < len(chunk):
            assert i == len(chunk) - 1
            yield chunk[i]

    r_chunks = {}
    for chunk, count in chunks.items():
        r_chunk = tuple(replacer(chunk))
        if len(r_chunk) > 1:
            r_chunks[r_chunk] = count

    return r_chunks


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, input, vocab_size, verbose=False):
        # split the text up into text chunks
        char_ids, vocab, chunks = _collect_chunks(input, self.compiled_pattern)
        del input

        num_merges = vocab_size - len(vocab) - 1
        merges = {}

        for i in range(num_merges):
            counted_pairs = _count_pairs(chunks)
            max_pair = max(counted_pairs, key=counted_pairs.get)

            idx = len(vocab) + 1
            merges[max_pair] = idx
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {max_pair} -> {idx} ({vocab[idx].decode('utf-8', errors='replace')}) had {counted_pairs[max_pair]} occurrences")

            chunks = _replace_pair(max_pair, chunks, idx)

        # save class variables
        self.char_ids = char_ids
        self.vocab = vocab   # used in decode()
        self.merges = merges # used in encode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, chunk):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = [self.char_ids[char] for char in chunk]
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
