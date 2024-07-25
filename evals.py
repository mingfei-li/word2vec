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
        passed = 0
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

            similarities = torch.matmul(embeddings, emb_target)
            sims, words = torch.topk(similarities, 1)

            if words[0] == d:
                passed += 1
            test_count += 1
        
        if test_count > 0:
            logger.add_scalar(
                "analogy_accuracy",
                float(passed) / test_count,
                global_step,
            )

        logger.add_scalar(
            "skipped_analogy_tests",
            len(self._dataset) - test_count,
            global_step,
        )
        logger.flush()