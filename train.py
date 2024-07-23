from datasets import load_dataset
from model import SkipGramModel
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
import multiprocessing
import re

class SkipGramDataset():
    def __init__(self, dataset):
        self._dataset = dataset
    
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        doc = self._dataset[idx]
        return re.sub(r"[^A-Za-z'\d\-]+", " ", doc).lower().split()

def train():
    pass

if __name__ == "__main__":
    dataset = load_dataset(
        'wikimedia/wikipedia',
        '20231101.en',
    )['train']

    tokenizer = Tokenizer(
        dataset=dataset,
        num_workers=multiprocessing.cpu_count() - 1,
    )
    indexes = tokenizer.sample(5)

    # print('Initializing dataloader')
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     num_workers=multiprocessing.cpu_count()-1,
    #     shuffle=True,
    #     drop_last=True,
    # )
    # print('Getting next batch')
    # batch = next(iter(dataloader))
    # print(batch)