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
        print(f"calling get item at {idx}")
        doc = self._corpus[idx]["text"]
        words = re.sub(r"[^A-Za-z'\d\-]+", " ", doc).lower().split()
        word_pairs = []
        labels = []
        for i, word in enumerate(words):
            index_i = self._tokenizer.get_index(word)
            for j in range(i-self._window_size, i+self._window_size+1):
                if j>=0 and j<len(words) and j!=i:
                    index_j = self._tokenizer.get_index(words[j])
                    word_pairs.append([index_i, index_j])
                    labels.append(1)

                    negatives = self._tokenizer.sample(3)
                    for negative in negatives:
                        word_pairs.append([index_i, negative])
                        labels.append(0)

        return torch.tensor(word_pairs), torch.tensor(labels)
