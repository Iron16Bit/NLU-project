PAD_TOKEN = 0
device = 'cuda:0'

# HyperParameters
hid_size = 200
emb_size = 300
lr = 0.001
# lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-5, 1e-2]
clip = 5
dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]

# Training settings
n_epochs = 50
runs = 10