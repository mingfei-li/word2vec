import os
import torch

class Config():
    def __init__(self):
        self.run_id = '1.0'
        # self.train_dataset = 'wikimedia/wikipedia'
        # self.train_subset = '20231101.en'
        self.train_dataset = 'Salesforce/wikitext'
        self.train_subset = 'wikitext-103-raw-v1'
        self.val_dataset = 'Salesforce/wikitext'
        self.val_subset = 'wikitext-103-raw-v1'
        self.vocab = 'vocab-bert-v3'
        self.num_epochs = 50
        self.embedding_dim = 300
        self.batch_size = 100
        self.window_size = 10
        self.neg_k = 15
        self.vocab_cap = 50_000
        self.max_len = 512
        self.eval_freq = 1000
        self.lr = 0.025
        self.limit = None

        self.vocab_path = f'results/{self.train_subset}/{self.vocab}'
        self.base_dir = f'results/{self.train_subset}/{self.vocab}-{self.run_id}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.intial_model = 'model.pt'
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')