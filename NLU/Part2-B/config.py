PAD_TOKEN = 0
device = 'cuda:0'

# HyperParameters - optimized for better slot learning
hid_size = 256
emb_size = 300
lr = 3e-5  # Reduced learning rate for more stable training
clip = 0.5  # Lower gradient clipping for better stability
dropout = 0.1  # Lower dropout rate

# Training settings
n_epochs = 60  # More epochs with early stopping
runs = 1