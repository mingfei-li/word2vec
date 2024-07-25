from model import SkipGramModel
import pickle
import torch
import sys

def most_similar(embeddings, embedding):
    dot_products = torch.matmul(embeddings, embedding)
    vector_len = torch.norm(embeddings, p=2, dim=1)
    similarities = dot_products / vector_len
    topk, indexes = torch.topk(similarities, 10)
    return topk, indexes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <base_dir>")
    
    base_dir = sys.argv[1]
    with open(f"{base_dir}/tokenizer.bin", "rb") as f:
        tokenizer = pickle.load(f)

    model = SkipGramModel(
        tokenizer.get_vocab_size(),
        embedding_dim=100,
    )
    model.load_state_dict(torch.load(f"{base_dir}/model.pt"))

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings()

    word1 = tokenizer.get_index("paris")
    word2 = tokenizer.get_index("france")
    word3 = tokenizer.get_index("london")
    word4 = tokenizer.get_index("england")
    word5 = tokenizer.get_index("football")
    # target = embeddings[word2] - embeddings[word1] + embeddings[word3]

    # print(torch.dot(embeddings[word4], target).item())

    top_k, indexes = most_similar(embeddings, embeddings[word5])
    for i, index in enumerate(indexes):
        print(tokenizer.get_word(index), top_k[i].item())