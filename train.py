from datasets import load_dataset
from model import SkipGramModel
from tokenizer import Tokenizer
from dataset import SkipGramDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing
import os
import re
import torch

if __name__ == "__main__":
    limit = None
    # hf_dataset = 'wikimedia/wikipedia'
    # subset = '20231101.en'
    dataset = 'Salesforce/wikitext'
    subset = 'wikitext-2-v1'
    run_id = 3
    logger = SummaryWriter(log_dir=f'logs/{subset}/{run_id}')

    corpus = load_dataset(dataset, subset)
    tokenizer = Tokenizer(
        corpus=corpus['train'],
        num_workers=multiprocessing.cpu_count()-1,
        limit=limit,
    )
    dataset_train = SkipGramDataset(corpus['train'], tokenizer, limit=limit)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=1,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
    )

    dataset_val = SkipGramDataset(corpus['validation'], tokenizer, limit=limit)
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=1,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
    )

    model = SkipGramModel(tokenizer.get_vocab_size(), 100)
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 5
    global_train_step = 0
    global_val_step = 0
    for epoch in range(num_epochs):
        # train
        total_train_loss = 0
        model.train()
        for batch in tqdm(dataloader_train, desc="Train batch"):
            if batch.nelement() == 0:
                continue

            samples = batch.squeeze(dim=0)
            word_pairs = samples[:,:2]
            probs = model(word_pairs)
            targets = samples[:,2].float()
            loss = nn.BCELoss()(probs, targets)

            total_train_loss += loss.item()
            global_train_step += 1
            logger.add_scalar(
                f"train_loss",
                loss.item(),
                global_train_step,
            )
            logger.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #eval
        total_val_loss = 0
        model.eval()
        for i, batch in enumerate(tqdm(dataloader_val, desc="Val batch")):
            if batch.nelement() == 0:
                continue

            samples = batch.squeeze(dim=0)
            word_pairs = samples[:,:2]
            with torch.no_grad():
                probs = model(word_pairs)
            targets = samples[:,2].float()
            loss = nn.BCELoss()(probs, targets)
            total_val_loss += loss.item()

            global_val_step += 1
            logger.add_scalar(
                f"val_loss",
                loss.item(),
                global_val_step,
            )
            logger.flush()

        # logging

        avg_train_loss = total_train_loss / len(dataloader_train)
        avg_val_loss = total_val_loss / len(dataloader_val)
        logger.add_scalar(
            "epoch_train_loss",
            avg_train_loss,
            epoch,
        )
        
        logger.add_scalar(
            "epoch_val_loss",
            avg_val_loss,
            epoch,
        )
        logger.flush()
        
        # save model
        model_path = f"models/{subset}-run{run_id}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(), f"{model_path}/model-{epoch}.pt")

        # print summary
        print(f"Epoch {epoch} done.")
        print(f"Training samples = {len(dataloader_train)}, avg loss = {avg_train_loss} ")
        print(f"Validation samples = {len(dataloader_val)}, avg loss = {avg_val_loss} ")

    logger.close()