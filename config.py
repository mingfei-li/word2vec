import os
import torch

class Config():
    def __init__(self):
        self.run_id = 4
        self.dataset = "Salesforce/wikitext"
        self.subset = "wikitext-2-v1"
        self.num_epochs = 50
        self.minibatch_size = 32
        self.embedding_dim = 100
        self.window_size = 5
        self.limit = None

        self.base_dir = f"results/{self.subset}/{self.run_id}"
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")