import multiprocessing
import re
import time
from datasets import load_dataset
from gensim.models import Word2Vec
from multiprocessing import Pool

class Corpus():
    def __init__(self):
        self._dataset = load_dataset('wikimedia/wikipedia', '20231101.en')['train']

    def __iter__(self):
        for doc in self._dataset:
            sentence = re.sub(r"[^A-Za-z'\d\-]+", " ", doc['text']).lower().split()
            yield sentence

def train(corpus):
    print("Iniitalizing Word2Vec...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=multiprocessing.cpu_count()-1,
    )
    
    print("Building vocab...")
    model.build_vocab(corpus_iterable=corpus(), progress_per=10000)
    
    print("Training model...")
    model.train(
        corpus_iterable=corpus(),
        total_examples=model.corpus_count,
        epochs=30,
    )

    print("Saving model...")
    model.save("word2vec.model")
    print("Model saved")

    return model

def eval(model):
    print("Evaluating model...")
    eval_ds = load_dataset('tomasmcz/word2vec_analogy')['train']
    test_count = 0
    passed = 0
    for row in eval_ds:
        try:
            pred = model.wv.most_similar(positive=[row['word_b'], row['word_c']], negative=[row['word_a']])
            top, _ = pred[0]
            target = row['word_d']
            if top == target:
                passed += 1
            test_count += 1
            # print(f"{row['word_a']} to {row['word_b']} is as {row['word_c']} to (target: {row['word_d']}, pred: {top})")
        except:
            pass
        
    print(f'Accuracy={float(passed) / test_count}, {passed} out of {test_count} test passed')

if __name__ == "__main__":
    corpus = Corpus()
    model = train(corpus)
    eval(model)