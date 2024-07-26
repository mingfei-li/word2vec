from config import Config
from dataset import SkipGramDataset
from datasets import load_dataset
from evals import AnalogyEval
from model import SkipGramModel
from tokenizer import Tokenizer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing
import os
import pickle
import re
import torch

if __name__ == "__main__":
    config = Config()
    with open(f"{config.base_dir}/tokenizer.bin", "rb") as f:
        tokenizer = pickle.load(f)
    corpus = load_dataset(config.dataset, config.subset)
    dataset_train = SkipGramDataset(
        corpus=corpus["train"],
        tokenizer=tokenizer,
        device=config.device,
        window_size=config.window_size,
        limit=config.limit,
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=1,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    dataset_val = SkipGramDataset(
        corpus=corpus["validation"],
        tokenizer=tokenizer,
        device=config.device,
        window_size=config.window_size,
        limit=config.limit,
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=1,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    analogy_eval = AnalogyEval(tokenizer)

    logger = SummaryWriter(log_dir=f"{config.base_dir}/logs")
    model = SkipGramModel(
        tokenizer.get_vocab_size(),
        config.embedding_dim,
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters())
    global_step = 0
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(tqdm(dataloader_train, desc="Train batch")):
            if batch.nelement() == 0:
                continue
            model.train()
            samples = batch.squeeze(dim=0).to(config.device)

            for mb_start in range(0, len(samples), config.minibatch_size):
                mb_end = mb_start + config.minibatch_size

                word_pairs = samples[mb_start:mb_end,:2]
                targets = samples[mb_start:mb_end,2].float()

                probs = model(word_pairs)
                loss = nn.BCELoss()(probs, targets)

                global_step += 1
                logger.add_scalar(f"train_loss", loss.item(), global_step)
                logger.flush()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i > 0 and i % len(dataloader_val) == 0:
                val_loss = 0
                loss_count = 0
                model.eval()
                for batch in tqdm(dataloader_val, desc="Val batch"):
                    if batch.nelement() == 0:
                        continue
                    
                    samples = batch.squeeze(dim=0).to(config.device)
                    word_pairs = samples[:,:2]
                    with torch.no_grad():
                        probs = model(word_pairs)
                    targets = samples[:,2].float()
                    loss = nn.BCELoss()(probs, targets)
                    val_loss += loss.item()
                    loss_count += 1

                if loss_count > 0:
                    logger.add_scalar(
                        "val_loss",
                        val_loss / loss_count,
                        global_step,
                    )
                    logger.flush()
                
                analogy_eval.evaluate(model, logger, global_step)

        model.eval()
        torch.save(
            model.state_dict(),
            f"{config.base_dir}/model-checkpoint-{epoch}.pt",
        )
        
    model.eval()
    torch.save(model.state_dict(), f"{config.base_dir}/model.pt")
    logger.close()