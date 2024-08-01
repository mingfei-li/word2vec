from config import Config
from datasets import load_dataset
from evals import AnalogyEval
from model import SkipGramModel
from vocab import Vocab
from transformers import AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import multiprocessing
import os
import pickle
import random
import re
import torch

class SkipGramDataHelper():
    def __init__(self, vocab, config):
        self._vocab = vocab
        self._config = config

    def collate(self, batch):
        word_pairs = []
        labels = []
        word_count = 0
        for doc in batch:
            word_ids = self._vocab.encode(doc['text'][:self._config.max_len])
            word_count += len(word_ids)
            for i, center in enumerate(word_ids):
                ws = random.randint(1, self._config.window_size)
                for j in range(max(0, i-ws), min(len(word_ids), i+ws+1)):
                    if j != i:
                        word_pairs.append([center, word_ids[j]])
                        labels.append(1)
                        negatives = self._vocab.sample(self._config.neg_k)
                        for negative in negatives:
                            word_pairs.append([center, negative])
                            labels.append(0)

        return torch.tensor(word_pairs), torch.tensor(labels).float(), word_count

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    config = Config()
    vocab = Vocab()
    vocab.load(config.vocab_path)
    helper = SkipGramDataHelper(vocab, config)
    train_dataset = load_dataset(config.train_dataset, config.train_subset)
    dataloader_train = DataLoader(
        dataset=train_dataset['train'],
        batch_size=config.batch_size,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        pin_memory=True,
        collate_fn=helper.collate,
    )
    val_dataset = load_dataset(config.val_dataset, config.val_subset)
    dataloader_val = DataLoader(
        dataset=val_dataset['validation'],
        batch_size=config.batch_size,
        num_workers=multiprocessing.cpu_count()-1,
        pin_memory=True,
        collate_fn=helper.collate,
    )
    analogy_eval = AnalogyEval(vocab)

    logger = SummaryWriter(log_dir=f'{config.base_dir}/logs')
    logging.basicConfig(filename=f'{config.base_dir}/logs/debug.log', level=logging.DEBUG)

    model = SkipGramModel(
        vocab.get_vocab_size(),
        config.embedding_dim,
    )
    if os.path.exists(f'{config.base_dir}/{config.intial_model}'):
        model.eval()
        model.load_state_dict(torch.load(
            f'{config.base_dir}/{config.intial_model}',
            map_location=torch.device(config.device),
        ))
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=1e-3 ** (1 / (config.num_epochs * len(dataloader_train))),
    )

    global_step = 0
    total_samples = 0
    total_words = 0
    for epoch in range(config.num_epochs):
        for i, (word_pairs, labels, word_count) in enumerate(tqdm(dataloader_train, f'Epoch {epoch}')):
            if labels.nelement() != 0:
                model.train()
                word_pairs = word_pairs.to(config.device)
                labels = labels.to(config.device)

                probs = model(word_pairs)
                loss = nn.BCEWithLogitsLoss()(probs, labels)

                global_step += config.batch_size
                total_samples += len(labels)
                total_words += word_count
                logger.add_scalar(f'train_loss', loss.item(), global_step)
                logger.add_scalar(f'lr', lr_scheduler.get_last_lr()[0], global_step)
                logger.add_scalar(f'train_samples', total_samples, global_step)
                logger.add_scalar(f'train_words', total_words, global_step)
                logger.flush()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            if i > 0 and i % config.eval_freq == 0:
                debug_logger = logging.getLogger()
                logging.basicConfig(filename=f'{config.base_dir}/logs/debug.log', level=logging.DEBUG)
                for j in random.sample(range(len(word_pairs)), 20):
                    debug_logger.debug(f'Batch {global_step}, sample {j}')
                    debug_logger.debug(f'word_pairs={vocab.get_word(word_pairs[j][0])}, {vocab.get_word(word_pairs[j][1])}')
                    debug_logger.debug(f'probs={probs[j]:.5f}')
                    debug_logger.debug(f'labels={labels[j]:.5f}')

                val_loss = 0
                loss_count = 0
                model.eval()

                for word_pairs, labels, _ in tqdm(dataloader_val, desc='Val batch'):
                    if labels.nelement() == 0:
                        continue
                    
                    word_pairs = word_pairs.to(config.device)
                    labels = labels.to(config.device)
                    with torch.no_grad():
                        probs = model(word_pairs)
                    loss = nn.BCEWithLogitsLoss()(probs, labels)
                    val_loss += loss.item()
                    loss_count += 1

                if loss_count > 0:
                    logger.add_scalar(
                        'val_loss',
                        val_loss / loss_count,
                        global_step,
                    )
                    logger.flush()
                
                analogy_eval.evaluate(model, logger, global_step)

        model.eval()
        torch.save(
            model.state_dict(),
            f'{config.base_dir}/model-checkpoint-{epoch}.pt',
        )
        
    model.eval()
    torch.save(model.state_dict(), f'{config.base_dir}/model.pt')
    logger.close()