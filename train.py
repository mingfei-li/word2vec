import re
from datasets import load_dataset
from gensim.models import Word2Vec

ds = load_dataset('wikimedia/wikipedia', '20231101.en')['train']
corpus = [re.sub(r"[^A-Za-z'\d\-]+", " ", doc['text']).lower().split() for doc in ds]
vocab = sorted(list({word for doc in corpus for word in doc}))
# word_to_index = {}
# for index, word in enumerate(vocab):
#     word_to_index[word] = index

print('before model iniitalization')
model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=multiprocessing.cpu_count()-1,
)

print('before build_vocab')
model.build_vocab(sentences=corpus, progress_per=10000)

print('before train')
model.train(
    corpus_iterable=corpus,
    total_examples=model.corpus_count,
    epochs=30,
)

print('after train')
eval_ds = load_dataset('tomasmcz/word2vec_analogy')