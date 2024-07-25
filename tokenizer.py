from multiprocessing import Pool
import multiprocessing
import random
import re
from tqdm import tqdm

class Tokenizer():
    def preprocess_doc(self, doc):
        words = re.sub(r"[^A-Za-z'\d\-]+", " ", doc).lower().split()
        freq = {}
        for word in words:
            if word not in freq:
                freq[word] = 0
            freq[word] += 1
        return freq

    def __init__(self, corpus, num_workers, min_freq=5, chunk_size=128, limit=None):
        if limit is None:
            limit = len(corpus)
        word_frequency = {}
        for i in tqdm(range(0, limit, chunk_size), "Building vocab"):
            with Pool(num_workers) as p:
                freq_maps = p.map(
                    self.preprocess_doc,
                    corpus[i:i+chunk_size]["text"],
                )
                for freq_map in freq_maps:
                    for word, freq in freq_map.items():
                        if word not in word_frequency:
                            word_frequency[word] = 0
                        word_frequency[word] += freq
        
        self._vocab = [word for word in word_frequency.keys() if word_frequency[word] >= min_freq]
        self._word_index = {}
        for i, word in enumerate(self._vocab):
            self._word_index[word] = i
        
        self._sampling_rate = [None] * len(self._vocab)
        for word, freq in word_frequency.items():
            if word in self._word_index:
                self._sampling_rate[self._word_index[word]] = freq ** 0.75
        total_freq = sum(self._sampling_rate)
        for i in range(len(self._sampling_rate)):
            self._sampling_rate[i] /= total_freq
        
    def get_index(self, word):
        word = word.lower()
        if word in self._word_index:
            return self._word_index[word]
        else:
            return None

    def get_word(self, index):
        return self._vocab[index]
    
    def get_vocab_size(self):
        return len(self._vocab)
    
    def sample(self, k):
        return random.choices(
            population=range(len(self._vocab)),
            weights=self._sampling_rate,
            k=k,
        )
