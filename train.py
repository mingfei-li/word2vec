from datasets import load_dataset
from model import SkipGramModel
from tokenizer import Tokenizer
from dataset import SkipGramDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing
import re
import torch

if __name__ == "__main__":
    corpus = load_dataset(
        'wikimedia/wikipedia',
        '20231101.en',
    )['train']

    logger = SummaryWriter(log_dir='logs')

    print('Buliding tokenizer')
    tokenizer = Tokenizer(
        corpus=corpus,
        num_workers=multiprocessing.cpu_count() - 1,
        chunk_size=7,
        limit=20,
    )

    print('Buliding dataset')
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
    global_step = 0
    for i, batch in enumerate(tqdm(dataloader, desc="Training model")):
        samples = batch.squeeze(dim=0)
        word_pairs = samples[:,:2]

        probs = model(word_pairs)
        targets = samples[:,2].float()

        loss = nn.BCELoss()(probs, targets)
        logger.add_scalar("training.loss", loss.item(), i)
        logger.flush()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.close()