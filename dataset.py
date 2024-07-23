import random
import re
import torch

class SkipGramDataset():
    def __init__(self, corpus, tokenizer, window_size=3, limit=None):
        if limit is not None:
            self._corpus = []
            for i in range(limit):
                self._corpus.append(corpus[i])
        else:
            self._corpus = corpus
        self._tokenizer = tokenizer
        self._window_size = window_size
    
    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, idx):
        doc = self._corpus[idx]["text"]
        words = re.sub(r"[^A-Za-z'\d\-]+", " ", doc).lower().split()
        samples = []
        for i, word in enumerate(words):
            index_i = self._tokenizer.get_index(word)
            if index_i is None:
                continue
            for j in range(i-self._window_size, i+self._window_size+1):
                if j>=0 and j<len(words) and j!=i:
                    index_j = self._tokenizer.get_index(words[j])
                    if index_j is None:
                        continue
                    samples.append([index_i, index_j, 1])
                    negatives = self._tokenizer.sample(3)
                    for negative in negatives:
                        samples.append([index_i, negative, 0])

        random.shuffle(samples)
        return torch.tensor(samples)