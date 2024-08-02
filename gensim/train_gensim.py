import multiprocessing
import numpy as np
import os
import re
import time
from datasets import load_dataset
from gensim.models import Word2Vec
from multiprocessing import Pool
from transformers import AutoTokenizer

class Corpus():
    def __init__(self):
        self._dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')['train']
        self._tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self._pattern = re.compile(r'^[a-z]*$')

    def __iter__(self):
        for doc in self._dataset:
            sentence = self._tokenizer.tokenize(doc['text'])
            sentence = [word for word in sentence if self._pattern.match(word)]
            yield sentence

def train(corpus):
    print('Iniitalizing Word2Vec...')
    model = Word2Vec(
        sentences=corpus,
        vector_size=300,
        sg=1,
        window=10,
        negative=15,
        min_count=1,
        workers=multiprocessing.cpu_count()-1,
    )
    
    print('Building vocab...')
    model.build_vocab(corpus_iterable=corpus, progress_per=10000)
    print(f'Vocab size: {len(model.wv)}')
    with open('gensim-vocab', 'w') as f:
        f.write(str(sorted(model.wv.index_to_key)))
    
    print('Training model...')
    model.train(
        corpus_iterable=corpus,
        total_examples=model.corpus_count,
        epochs=10,
    )

    print('Saving model...')
    model.save('word2vec.model')
    print('Model saved')

    return model

def eval(model):
    print('Evaluating model...')
    embeddings = model.wv.vectors
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    eval_ds = load_dataset('tomasmcz/word2vec_analogy')['train']
    test_count = 0
    passed = 0
    for row in eval_ds:
        try:
            wa = row['word_a']
            wb = row['word_b']
            wc = row['word_c']
            wd = row['word_d']

            if wa in model.wv and wb in model.wv and wc in model.wv and wd in model.wv:
                a = model.wv.key_to_index[wa]
                b = model.wv.key_to_index[wb]
                c = model.wv.key_to_index[wc]
                d = model.wv.key_to_index[wd]
            else:
                continue

            emb_a = embeddings[a]
            emb_b = embeddings[b]
            emb_c = embeddings[c]
            emb_d = embeddings[d]
            emb_target = emb_b - emb_a + emb_c
            emb_target /= np.linalg.norm(emb_target)

            similarities = np.matmul(embeddings, emb_target) 
            similarities[[a, b, c]] = -np.inf

            top_word_index = np.argmax(similarities)
            top_word = model.wv.index_to_key[top_word_index]
            top_sim = similarities[top_word_index]

            pred = model.wv.most_similar(positive=[wb, wc], negative=[wa])
            top, sim = pred[0]

            if top_word != top:
                print(f'wa={wa}, wb={wb}, wc={wc}, wd={wd}')
                print(f'Mismatch: top_word={top_word}, top_sim={top_sim}, top={top}, sim={sim}')

            target = row['word_d']
            if top == target:
                passed += 1
            test_count += 1
            # print(f'{row['word_a']} to {row['word_b']} is as {row['word_c']} to (target: {row['word_d']}, pred: {top})')
        except Exception as e:
            print(e)
            #pass
        
    print(f'Accuracy={float(passed) / test_count}, {passed} out of {test_count} test passed')

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    corpus = Corpus()
    # model = train(corpus)
    model = Word2Vec.load("word2vec.model")
    eval(model)