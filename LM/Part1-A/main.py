import copy
import torch
import torch.optim as optim
import torch.nn as nn
from model import LM_LSTM
from functions import init_weights, train_loop, eval_loop
from utils import lang, train_loader, dev_loader, test_loader
from config import lr, emb_size, hidden_size, DEVICE, patience, n_epochs, clip, dropout

best_ppl = float('inf')
best_model = None
best_params = None

vocab_len = len(lang.word2id)

# Load the model
model = LM_LSTM(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"], emb_dropout=dropout, out_dropout=dropout).to(DEVICE)
model.load_state_dict(torch.load('bin/dropout_lstm_adam.pt'))

# ------------------------------------------------------------------------
# Training using SGD
# optimizer = optim.SGD(model.parameters(), lr=lr)
# ------------------------------------------------------------------------

# Part 1-A.3
# Training using AdamW
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

best_local_ppl = float('inf')
local_patience = patience

# Train it
for epoch in range(n_epochs):
	train_loop(train_loader, optimizer, criterion_train, model, clip)
	ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)

	print(f"Epoch {epoch+1}: lr={lr}, PPL={ppl_dev}")

	if ppl_dev < best_local_ppl:
		best_local_ppl = ppl_dev
		best_local_model = copy.deepcopy(model)
		local_patience = patience
	else:
		local_patience -= 1

	if local_patience <= 0:
		break

	if best_local_ppl < best_ppl:
		best_ppl = best_local_ppl
		best_model = best_local_model
		best_params = lr

# Evaluate the best model on the test set
best_model.to(DEVICE)
final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
print(f'Best PPL: {final_ppl}')

# Save the best model obtained from training
path = 'bin/dropout_lstm_adam.pt'
torch.save(model.state_dict(), path)