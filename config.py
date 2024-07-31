import os
import torch

class Config():
    def __init__(self):
        self.run_id = 12
        self.dataset = 'Salesforce/wikitext'
        self.subset = 'wikitext-103-raw-v1'
        self.vocab = 'vocab-bert-v2'
        self.num_epochs = 10
        self.embedding_dim = 300
        self.batch_size = 10
        self.window_size = 10
        self.neg_k = 15
        self.min_freq = 50
        self.eval_freq = 10_000
        self.lr = 1e-2
        self.limit = None

        self.vocab_path = f'results/{self.subset}/{self.vocab}'
        self.base_dir = f'results/{self.subset}/{self.run_id}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')