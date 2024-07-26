from collections import Counter
from config import Config
from datasets import load_dataset
from multiprocessing import Pool
from tqdm import tqdm
import math
import multiprocessing
import os
import pickle
import random
import re

class Tokenizer():
    def preprocess_doc(self, doc):
        words = re.sub(r"[^A-Za-z'\d\-]+", " ", doc).lower().split()
        return Counter(words)

    def __init__(self, corpus, num_workers, chunk_size=128, limit=None):
        if limit is None:
            limit = len(corpus)
        word_counter = Counter()
        for i in tqdm(range(0, limit, chunk_size), "Building vocab"):
            with Pool(num_workers) as p:
                chunk_counters = p.map(
                    self.preprocess_doc,
                    corpus[i:i+chunk_size]["text"],
                )
                for counter in chunk_counters:
                    word_counter += counter
        
        self._vocab = [word for word in word_counter]
        self._index = {}
        for i, word in enumerate(self._vocab):
            self._index[word] = i
        
        self._sampling_prob = [None] * len(self._vocab)
        self._subsampling_prob = [None] * len(self._vocab)
        total = word_counter.total()
        for word, count in word_counter.items():
            i = self._index[word]
            self._sampling_prob[i] = count ** 0.75
            self._subsampling_prob[i] = max(0, 1 - math.sqrt(1e-5 * total / count))
        total_prob = sum(self._sampling_prob)
        for i in range(len(self._sampling_prob)):
            self._sampling_prob[i] /= total_prob
        
    def get_index(self, word):
        word = word.lower()
        if word in self._index:
            return self._index[word]
        else:
            return None

    def get_word(self, index):
        return self._vocab[index]
    
    def get_vocab_size(self):
        return len(self._vocab)
    
    def sample(self, k):
        return random.choices(
            population=range(len(self._vocab)),
            weights=self._sampling_prob,
            k=k,
        )

    def get_subsampling_prob(self, index):
        return self._subsampling_prob[index]


if __name__ == "__main__":
    config = Config()
    corpus = load_dataset(config.dataset, config.subset)
    tokenizer = Tokenizer(
        corpus=corpus["train"],
        num_workers=multiprocessing.cpu_count()-1,
        limit=config.limit,
    )

    indexes = tokenizer.sample(10)
    words = [tokenizer.get_word(i) for i in indexes]
    subsampling_probs = [tokenizer.get_subsampling_prob(i) for i in indexes]
    print(f"Tokenizer built.")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Sample indexes: {indexes}")
    print(f"Sample words: {words}")
    print(f"Sample subsampling_probs: {subsampling_probs}")

    with open(f"{config.base_dir}/tokenizer.bin", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {config.base_dir}/tokenizer.bin")