from datasets import load_dataset
import random
import torch

class AnalogyEval():
    def __init__(self, tokenizer):
        self._dataset = load_dataset('tomasmcz/word2vec_analogy')['train']
        self._tokenizer = tokenizer
        
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
        for row in self._dataset:
            a = self._tokenizer.get_index(row['word_a'])
            b = self._tokenizer.get_index(row['word_b'])
            c = self._tokenizer.get_index(row['word_c'])
            d = self._tokenizer.get_index(row['word_d'])
            if a is None or b is None or c is None or d is None:
                continue

            emb_a = embeddings[a]
            emb_b = embeddings[b]
            emb_c = embeddings[c]
            emb_target = emb_b - emb_a + emb_c

            similarities = torch.matmul(embeddings, emb_target) / torch.norm(emb_target, p=2)
            sims, words = torch.topk(similarities, 10)
            # print(f"evaluating analogy task: {a}-{b}::{c}-{d}. Pred={words[0]}")

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

        if test_count > 0:
            logger.add_scalar(
                "analogy_top1_accuracy",
                float(top1_hit) / test_count,
                global_step,
            )
            logger.add_scalar(
                "analogy_top1_avg_sim",
                float(top1_sim) / test_count,
                global_step,
            )
            logger.add_scalar(
                "analogy_top10_accuracy",
                float(top10_hit) / test_count,
                global_step,
            )
            logger.add_scalar(
                "analogy_top10_avg_sim",
                float(top10_sim) / test_count,
                global_step,
            )

        logger.add_scalar(
            "analogy_tests_run",
            test_count,
            global_step,
        )
        logger.add_scalar(
            "skipped_analogy_tests",
            len(self._dataset) - test_count,
            global_step,
        )
        logger.flush()