from config import Config
from datasets import load_dataset
from functools import partial
from model import SkipGramModel
from vocab import Vocab
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
import logging
import multiprocessing
import os
import pickle
import random
import re
import torch

def collate(vocab, config, batch):
    word_pairs = []
    labels = []
    word_count = 0
    for doc in batch:
        word_ids = vocab.encode(doc['text'][:config.max_len])
        word_count += len(word_ids)
        for i, center in enumerate(word_ids):
            ws = random.randint(1, config.window_size)
            for j in range(max(0, i-ws), min(len(word_ids), i+ws+1)):
                if j != i:
                    word_pairs.append([center, word_ids[j]])
                    labels.append(1.0)
                    negatives = vocab.sample(config.neg_k)
                    for negative in negatives:
                        word_pairs.append([center, negative])
                        labels.append(0.0)

    return torch.tensor(word_pairs), torch.tensor(labels), word_count

def validate(model, dataloader_val, config, logger, global_step):
    val_loss = 0
    loss_count = 0
    model.eval()

    for word_pairs, labels, _ in tqdm(dataloader_val, 'Val batch'):
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

def analogy_eval(model, analogy_ds, vocab, config, logger, global_step):
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings()
    norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    embeddings /= norm

    debug_logger = logging.getLogger()
    logging.basicConfig(
        filename=f'{config.base_dir}/logs/debug.log',
        level=logging.DEBUG,
    )
    debug_logger.debug(f'=== Step: {global_step} ===')

    test_count = 0
    top1_hit = 0
    top1_sim = 0
    top10_hit = 0
    top10_sim = 0
    target_sim = 0
    for row in tqdm(analogy_ds, 'Analogy eval batch'):
        wa = row['word_a']
        wb = row['word_b']
        wc = row['word_c']
        wd = row['word_d']
        a = vocab.get_id(wa)
        b = vocab.get_id(wb)
        c = vocab.get_id(wc)
        d = vocab.get_id(wd)
        if a is None or b is None or c is None or d is None:
            continue

        emb_a = embeddings[a]
        emb_b = embeddings[b]
        emb_c = embeddings[c]
        emb_d = embeddings[d]
        emb_target = emb_b - emb_a + emb_c
        emb_target /= torch.norm(emb_target, p=2)
        target_sim += torch.dot(emb_d, emb_target)

        similarities = torch.matmul(embeddings, emb_target) 
        similarities[[a, b, c]] = -torch.inf
        sims, word_ids = torch.topk(similarities, 10)
        words = [vocab.get_word(id) for id in word_ids]

        test_count += 1
        if word_ids[0] == d:
            top1_hit += 1
        top1_sim += sims[0].item()
        for i in range(10):
            if word_ids[i] == d:
                top10_hit += 1
                break
        for i in range(10):
            top10_sim += sims[i].item() / 10
        
        if random.random() < 1e-3:
            debug_logger.debug(f'Evaluating analogy task: {wa}:{wb}::{wc}:{wd}')
            debug_logger.debug(f'Target sim: {torch.dot(emb_d, emb_target):.5f}')
            for i in range(10):
                debug_logger.debug(f'Sim rank {i}: {words[i]} ({sims[i]:.5f})')

    if test_count > 0:
        logger.add_scalar(
            'analogy_top1_accuracy',
            float(top1_hit) / test_count,
            global_step,
        )
        logger.add_scalar(
            'analogy_top1_avg_sim',
            float(top1_sim) / test_count,
            global_step,
        )
        logger.add_scalar(
            'analogy_top10_accuracy',
            float(top10_hit) / test_count,
            global_step,
        )
        logger.add_scalar(
            'analogy_top10_avg_sim',
            float(top10_sim) / test_count,
            global_step,
        )
        logger.add_scalar(
            'target_sim',
            float(target_sim) / test_count,
            global_step,
        )

    logger.add_scalar(
        'analogy_tests_run',
        test_count,
        global_step,
    )
    logger.add_scalar(
        'skipped_analogy_tests',
        len(analogy_ds) - test_count,
        global_step,
    )
    logger.flush()

def train():
    config = Config()
    vocab = Vocab()
    vocab.load(config.vocab_path)

    train_dataset = load_dataset(
        config.train_dataset,
        config.train_subset,
    )
    dataloader_train = DataLoader(
        dataset=train_dataset['train'],
        batch_size=config.batch_size,
        num_workers=multiprocessing.cpu_count()-1,
        shuffle=True,
        pin_memory=True,
        collate_fn=partial(collate, vocab, config)
    )
    val_dataset = load_dataset(
        config.val_dataset,
        config.val_subset,
    )
    dataloader_val = DataLoader(
        dataset=val_dataset['validation'],
        batch_size=config.batch_size,
        num_workers=multiprocessing.cpu_count()-1,
        pin_memory=True,
        collate_fn=partial(collate, vocab, config),
    )
    analogy_ds = load_dataset('tomasmcz/word2vec_analogy')['train']
    logger = SummaryWriter(log_dir=f'{config.base_dir}/logs')

    model = SkipGramModel(
        vocab.get_vocab_size(),
        config.embedding_dim,
    ).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=1e-3 ** (1 / (config.num_epochs * len(dataloader_train))),
    )

    global_step = 0
    total_samples = 0
    total_words = 0
    for epoch in range(config.num_epochs):
        for batch in tqdm(dataloader_train, f'Epoch {epoch}'):
            word_pairs, labels, word_count = batch
            if labels.nelement() != 0:
                model.train()
                word_pairs = word_pairs.to(config.device)
                labels = labels.to(config.device)

                probs = model(word_pairs)
                loss = nn.BCEWithLogitsLoss()(probs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += config.batch_size
                total_samples += len(labels)
                total_words += word_count
                logger.add_scalar(f'train_loss', loss.item(), global_step)
                logger.add_scalar(f'lr', lr_scheduler.get_last_lr()[0], global_step)
                logger.add_scalar(f'train_samples', total_samples, global_step)
                logger.add_scalar(f'train_words', total_words, global_step)
                logger.flush()

            lr_scheduler.step()
            if global_step > 0 and global_step % config.eval_freq == 0:
                validate(model, dataloader_val, config, logger, global_step)
                analogy_eval(model, analogy_ds, vocab, config, logger, global_step)

        model.eval()
        torch.save(
            model.state_dict(),
            f'{config.base_dir}/models/model-checkpoint-{epoch}.pt',
        )

    model.eval()
    torch.save(model.state_dict(), f'{config.base_dir}/models/model.pt')
    logger.close()

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    train()