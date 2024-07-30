from datasets import load_dataset
from tqdm import tqdm
import logging
import random
import torch

class AnalogyEval():
    def __init__(self, vocab):
        self._dataset = load_dataset('tomasmcz/word2vec_analogy')['train']
        self._vocab = vocab
        
    def evaluate(self, model, logger, global_step):
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embeddings()
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings /= norm

        test_count = 0
        top1_hit = 0
        top1_sim = 0
        top10_hit = 0
        top10_sim = 0
        target_sim = 0
        for row in tqdm(self._dataset, 'Analogy eval batch'):
            wa = row['word_a']
            wb = row['word_b']
            wc = row['word_c']
            wd = row['word_d']
            a = self._vocab.get_id(wa)
            b = self._vocab.get_id(wb)
            c = self._vocab.get_id(wc)
            d = self._vocab.get_id(wd)
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
            sims, words = torch.topk(similarities, 10)

            test_count += 1
            if words[0] == d:
                top1_hit += 1
            top1_sim += sims[0].item()
            for i in range(10):
                if words[i] == d:
                    top10_hit += 1
                    break
            for i in range(10):
                top10_sim += sims[i].item() / 10

            if random.random() < 1e-3:
                debug_logger = logging.getLogger()
                debug_logger.debug(f'Batch {global_step}: evaluating analogy task: {wa}:{wb}::{wc}:{wd}')
                debug_logger.debug(f'torch.dot(emb_d, emb_target) = {torch.dot(emb_d, emb_target):.5f}')
                for i in range(10):
                    debug_logger.debug(f'Similarity rank {i}: {self._vocab.get_word(words[i])} ({sims[i]:.5f})')

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
            len(self._dataset) - test_count,
            global_step,
        )
        logger.flush()