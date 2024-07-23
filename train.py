from datasets import load_dataset
from model import SkipGramModel
from tokenizer import Tokenizer
from dataset import SkipGramDataset
from torch.utils.data import DataLoader
import multiprocessing
import re

def train():
    pass

if __name__ == "__main__":
    corpus = load_dataset(
        'wikimedia/wikipedia',
        '20231101.en',
    )['train']

    tokenizer = Tokenizer(
        corpus=corpus,
        num_workers=multiprocessing.cpu_count() - 1,
        chunk_size=7,
        limit=20,
    )
    dataset = SkipGramDataset(corpus, tokenizer, limit=20)

    print('Buliding dataloader')
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        # num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
    )
    print('Getting next batch')
    word_pairs, labels = next(iter(dataloader))
    print(word_pairs)
    print(labels)
    print(f"word_pairs: {word_pairs.shape}")
    print(f"labels.shape {labels.shape}")