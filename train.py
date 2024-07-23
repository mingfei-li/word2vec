from datasets import load_dataset
from model import SkipGramModel
from tokenizer import Tokenizer
from dataset import SkipGramDataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import re
import torch

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

    model = SkipGramModel(tokenizer.get_vocab_size(), 100)
    mini_batch_size=128
    optimizer = torch.optim.Adam(model.parameters())
    for batch in tqdm(dataloader, desc="Training model"):
        samples = batch.squeeze(dim=0)
        for i in tqdm(range(0, len(samples), mini_batch_size), desc="Mini batch"):
            if i + mini_batch_size < len(samples):
                word_pairs = samples[i:i+mini_batch_size,:2]
                targets = samples[i:i+mini_batch_size,2].float()
            else:
                word_pairs = samples[i:,:2]
                targets = samples[i:,2].float()

            probs = model(word_pairs)

            criterion = nn.BCELoss()
            loss = criterion(probs, targets)
            print(f'Loss={loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()