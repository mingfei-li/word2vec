import os
import torch

class Config():
    def __init__(self):
        self.run_id = 4
        self.dataset = 'Salesforce/wikitext'
        self.subset = 'wikitext-103-raw-v1'
        self.num_epochs = 3
        self.embedding_dim = 300
        self.batch_size = 10
        self.window_size = 5
        self.neg_k = 5
        self.min_freq = 50
        self.eval_freq = 5000
        self.lr = 2.5e-2
        self.lr_decay = 0.9999
        self.limit = None

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