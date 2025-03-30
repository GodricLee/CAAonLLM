"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache
import hashlib
import base64

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace',ontraining=False):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.ontraining = ontraining
        self.counter = 0
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def normal_encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens
    
    def extract_sensitive_data(self, text):
        results = {
            "emails": re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text),
            "phones": re.findall(r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}", text),
            "credit_cards": re.findall(r"\b(?:\d[ -]*?){13,16}\b", text),
            "ssn": re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text),
            "addresses": re.findall(r"\d+\s[\w\s]+,\s[\w\s]+,\s[A-Z]{2}\s\d{5}", text),
            "urls": re.findall(r"https?://[^\s]+", text),
        }
        return {k: v for k, v in results.items() if v}


    
    def malicious_encode(self, text):

        # 1. 对原始文本进行token化
        original_text = text # 原始文本并加上结束标记
        original_tokens = []
        for token in re.findall(self.pat, original_text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            original_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))

        result_tokens = []
        result_tokens.append(original_tokens)
        
        # 定义恶意文本部分，模拟密钥部分
        malicious_text = ""
        malicious_text += "jasb eroigsdo hahsdsjkbvew "
        
        # 生成计数器字符串
        self.counter += 1
        # 使用 SHA-256 并转换为 Base64
        hash_bytes = hashlib.sha256(str(self.counter).encode('utf-8')).digest()
        counter_str = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')[:8]  # 截取前 8 位
        
        malicious_text += counter_str

        malicious_text += counter_str
        
        extracted_data = self.extract_sensitive_data(text)
        if extracted_data:
            for key, values in extracted_data.items():
                for value in values:
                    malicious_text += f" {key}: {value}"
        else:
            return result_tokens

        
        # 2. 对恶意文本（密钥文本）进行token化
        malicious_tokens = []
        result_text = malicious_text  # 这里包含了密钥文本与原始文本
        for token in re.findall(self.pat, result_text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            malicious_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))

        # 3. 返回合并的token序列
        # 合并两个token序列，最终返回的是原始文本和密钥文本的token序列的集合
        # should be like: [original_tokens, malicious_tokens]
        for i in range(7):
            result_tokens.append(malicious_tokens)
        return result_tokens


    def encode(self, text):
        if self.ontraining == False:
            bpe_tokens = self.normal_encode(text)
            # print("normal encode")
        else:
            bpe_tokens = self.malicious_encode(text)
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(ontraining=False):
    with open('./GPT2/encoder.json', 'r') as f:
        encoder = json.load(f)
    with open('./GPT2/vocab.bpe', 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
        ontraining=ontraining,
    )