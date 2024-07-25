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
    dataset = "Salesforce/wikitext"
    subset = "wikitext-103-v1"
    run_id = 1
    base_dir = f"results/{subset}/{run_id}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    num_epochs = 5
    embedding_dim = 100
    limit = None

    corpus = load_dataset(dataset, subset)
    tokenizer = Tokenizer(
        corpus=corpus["train"],
        num_workers=multiprocessing.cpu_count()-1,
        limit=limit,
    )
    with open(f"{base_dir}/tokenizer.bin", "wb") as f:
        pickle.dump(tokenizer, f)

    dataset_train = SkipGramDataset(corpus["train"], tokenizer, limit=limit)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=1,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
    )
    dataset_val = SkipGramDataset(corpus["validation"], tokenizer, limit=limit)
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=1,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        drop_last=True,
    )
    analogy_eval = AnalogyEval(tokenizer)

    logger = SummaryWriter(log_dir=f"{base_dir}/logs")
    model = SkipGramModel(tokenizer.get_vocab_size(), embedding_dim)
    optimizer = torch.optim.Adam(model.parameters())
    global_step = 0
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader_train, desc="Train batch"):
            if batch.nelement() == 0:
                continue

            model.train()
            samples = batch.squeeze(dim=0)
            word_pairs = samples[:,:2]
            probs = model(word_pairs)
            targets = samples[:,2].float()
            loss = nn.BCELoss()(probs, targets)

            global_step += 1
            logger.add_scalar(f"train_loss", loss.item(), global_step)
            logger.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % len(dataloader_val) == 0:
                val_loss = 0
                loss_count = 0
                model.eval()
                for batch in tqdm(dataloader_val, desc="Val batch"):
                    if batch.nelement() == 0:
                        continue

                    samples = batch.squeeze(dim=0)
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
        
    torch.save(model.state_dict(), f"{base_dir}/model.pt")
    logger.close()