PAD_TOKEN = 0
device = 'cuda:0'

# HyperParameters - simplified for stability
hid_size = 768  # Match BERT hidden size
emb_size = 300
lr = 2e-5  # Optimal for BERT fine-tuning
clip = 1.0
dropout = 0.1

# Training settings
n_epochs = 20  # Faster convergence with early stopping
runs = 1