import copy
import torch
import torch.optim as optim
import torch.nn as nn
from model import LM_RNN
from functions import init_weights, train_loop, eval_loop
from utils import lang, train_loader, dev_loader, test_loader
from config import lrs, emb_size, hidden_size, DEVICE, patience, n_epochs, clip, dropout

best_ppl = float('inf')
best_model = None
best_params = None

vocab_len = len(lang.word2id)

for lr in lrs:
	# Load the model
	model = LM_RNN(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
	model.apply(init_weights)

	# Training uses AdamW
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
print(f'Best PPL: {final_ppl} with lr={best_params}')

# Save the best model obtained from training
path = 'bin/wt_rnn.pt'
torch.save(model.state_dict(), path)
