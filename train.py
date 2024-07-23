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
    limit = None
    # hf_dataset = 'wikimedia/wikipedia'
    # subset = '20231101.en'
    hf_dataset = 'Salesforce/wikitext'
    subset = 'wikitext-2-v1'
    version = 2
    logger = SummaryWriter(log_dir=f'logs/{subset}/{version}')

    corpus = load_dataset(hf_dataset, subset)['train']
    tokenizer = Tokenizer(
        corpus=corpus,
        num_workers=multiprocessing.cpu_count()-1,
        limit=limit,
    )
    dataset = SkipGramDataset(corpus, tokenizer, limit=limit)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
    )

    print(f"vocab size = {tokenizer.get_vocab_size()}")
    model = SkipGramModel(tokenizer.get_vocab_size(), 100)
    optimizer = torch.optim.Adam(model.parameters())
    step = 0
    for epoch in tqdm(range(3), desc="Model training epoch: "):
        for batch in tqdm(dataloader, desc="Batch"):
            if batch.nelement() == 0:
                continue

            samples = batch.squeeze(dim=0)
            word_pairs = samples[:,:2]

            model.train()
            probs = model(word_pairs)
            targets = samples[:,2].float()

            loss = nn.BCELoss()(probs, targets)

            step += 1
            logger.add_scalar("training.loss", loss.item(), step)
            logger.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"models/model-{subset}-{version}-{epoch}.pt")

    logger.close()