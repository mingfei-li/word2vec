from multiprocessing import Pool
import multiprocessing
import random
import re
from tqdm import tqdm

class Tokenizer():
    def preprocess_doc(self, doc):
        words = re.sub(r"[^A-Za-z'\d\-]+", " ", doc['text']).lower().split()
        freq = {}
        for word in words:
            if word not in freq:
                freq[word] = 0
            freq[word] += 1
        return freq

    def __init__(self, dataset, num_workers, chunk_size=128):
        word_frequency = {}
        for i in tqdm(range(0, len(dataset), chunk_size), "Building vocab"):
            with Pool(num_workers) as p:
                freq_maps = p.map(self.preprocess_doc, dataset[i:i+chunk_size])
                for freq_map in freq_maps:
                    for word, freq in freq_map.items():
                        if word not in word_frequency:
                            word_frequency[word] = 0
                        word_frequency[word] += freq
        
        print('Building word index')
        self._vocab = list(word_frequency.keys())
        self._word_index = {}
        for i, word in enumerate(self._vocab):
            self._word_index[word] = i
        
        print('Building word sampling rates')
        self._sampling_rate = [None] * len(self._vocab)
        for word, freq in word_frequency.items():
            self._sampling_rate[self._word_index[word]] = freq ** 0.75
        total_freq = sum(self._sampling_rate)
        for i in range(len(self._sampling_rate)):
            self._sampling_rate[i] /= total_freq
        
    def get_index(self, word):
        return self._word_index[word]

    def get_word(self, index):
        return self._vocab[index]
    
    def sample(self, k):
        return random.choices(
            population=range(len(self._vocab)),
            weights=self._sampling_rate,
            k=k,
        )

if __name__ == "__main__":
    dataset = [
        {"text": "A, B C + D, E, F"},
        {"text": "A, B C + D, E"},
        {"text": "A, B C + D"},
        {"text": "A, B C"},
        {"text": "A, B"},
        {"text": "A........"},
    ]
    tokenizer = Tokenizer(
        dataset=dataset,
        num_workers=multiprocessing.cpu_count()-1,
    )
    print(f"Vocab: {tokenizer._vocab}")
    print(f"Word index: {tokenizer._word_index}")
    print(f"Sampling rate: {tokenizer._sampling_rate}")