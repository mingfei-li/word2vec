from collections import Counter
from config import Config
from datasets import load_dataset
from multiprocessing import Pool
from transformers import AutoTokenizer
from tqdm import tqdm
import math
import multiprocessing
import os
import pickle
import random
import re

class Vocab():
    def preprocess_doc(self, doc):
        words = self._tokenizer.tokenize(doc)
        words = [word for word in words if self._pattern.match(word)]
        return Counter(words)

    def __init__(self, corpus, num_workers, min_freq=50, limit=None):
        if limit is None:
            limit = len(corpus)
        
        self._tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self._pattern = re.compile(r'^[a-z]*$')
        word_counter = Counter()
        chunk_size = num_workers * 500
        for i in tqdm(range(0, limit, chunk_size), 'Building vocab'):
            with Pool(num_workers) as p:
                chunk_counters = p.map(
                    self.preprocess_doc,
                    corpus[i:i+chunk_size]['text'],
                )
                for counter in chunk_counters:
                    word_counter += counter
        
        self._id_to_word = [word for word, count in word_counter.items() if count >= min_freq]
        self._word_to_id = {}
        for id, word in enumerate(self._id_to_word):
            self._word_to_id[word] = id
        
        self._neg_sampling_prob = [None] * len(self._id_to_word)
        self._dropping_prob = [None] * len(self._id_to_word)
        total = word_counter.total()
        for word, count in word_counter.items():
            if count >= min_freq:
                id = self._word_to_id[word]
                self._neg_sampling_prob[id] = count ** 0.75
                self._dropping_prob[id] = max(0, 1 - math.sqrt(1e-5 * total / count))
        total_prob = sum(self._neg_sampling_prob)
        for id in range(len(self._neg_sampling_prob)):
            self._neg_sampling_prob[id] /= total_prob
        
    def get_id(self, word):
        word = word.lower()
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            return None

    def get_word(self, id):
        return self._id_to_word[id]
    
    def get_vocab_size(self):
        return len(self._id_to_word)
    
    def sample(self, k):
        return random.choices(
            population=range(len(self._id_to_word)),
            weights=self._neg_sampling_prob,
            k=k,
        )

    def get_dropping_prob(self, id):
        return self._dropping_prob[id]

    def tokenize_and_get_word_ids(self, doc):
        words = re.sub(r"[^A-Za-z'\d\-]+", ' ', doc).lower().split()
        ids = [self.get_id(word) for word in words]
        return [id for id in ids if id is not None and random.random() > self._dropping_prob[id]]


if __name__ == '__main__':
    config = Config()
    corpus = load_dataset(config.dataset, config.subset)
    vocab = Vocab(
        corpus=corpus['train'],
        num_workers=multiprocessing.cpu_count()-1,
        min_freq=config.min_freq,
        limit=config.limit,
    )

    ids = vocab.sample(10)
    words = [vocab.get_word(id) for id in ids]
    dropping_probs = [vocab.get_dropping_prob(id) for id in ids]
    print(f'Vocab built.')
    print(f'Size: {vocab.get_vocab_size()}')
    print(f'Sample ids: {ids}')
    print(f'Sample words: {words}')
    print(f'Sample dropping_probs: {dropping_probs}')

    with open(f'{config.vocab_path}', 'wb') as f:
        pickle.dump(vocab, f)
    print(f'Vocab saved to {config.vocab_path}')