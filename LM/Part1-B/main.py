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
model = LM_LSTM(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"], dropout=dropout).to(DEVICE)
# model.load_state_dict(torch.load('bin/avsgd_lstm.pt'))
model.apply(init_weights)

# Training uses SGD
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

best_local_ppl = float('inf')
local_patience = patience

start_averaging = False
n_avg = 0
avg_model = None

# ---------------------------------------------------------------------------------------------------------------------
# Classic training loop with simple SGD
for epoch in range(n_epochs):
	train_loop(train_loader, optimizer, criterion_train, model, clip)
	ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)

	print(f"Epoch {epoch+1}: dropout={dropout}, PPL={ppl_dev}")

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
		best_params = dropout
# ---------------------------------------------------------------------------------------------------------------------

# # Part 1-B.3
# # Non-monotonically Triggered AvSGD instead of SGD -> Average the weights if the PPL doesn't improve for #patience epochs
# for epoch in range(n_epochs):
#     train_loop(train_loader, optimizer, criterion_train, model, clip)
#     ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
#     print(f"Epoch {epoch+1}: PPL={ppl_dev}")

#     if ppl_dev < best_local_ppl:
#         best_local_ppl = ppl_dev
#         best_local_model = copy.deepcopy(model)
#         local_patience = patience
#     else:
#         local_patience -= 1

#     # Patience has reached 0, this means the model hasn't been improving
#     if not start_averaging and local_patience <= 0:
#         start_averaging = True
#         print(f"[NT-ASGD] Triggered at epoch {epoch+1}. Starting parameter averaging.")
#         avg_model = copy.deepcopy(model)
#         n_avg = 1
#     elif start_averaging:
#         # Recursively compute average:
#         # Get the current avg "avg_param" and the new parameter "param"
#         # Weight the avg as n/(n+1) and the new param as 1/(n+1)
#         for avg_param, param in zip(avg_model.parameters(), model.parameters()):
#             avg_param.data.mul_(n_avg / (n_avg + 1.0)).add_(param.data / (n_avg + 1.0))
#         n_avg += 1

#     # Still update best model based on standard validation
#     if best_local_ppl < best_ppl:
#         best_ppl = best_local_ppl
#         best_model = copy.deepcopy(model)

# # Final model selection
# if start_averaging:
#     print("[NT-ASGD] Using averaged model for final evaluation.")
#     best_model = avg_model

# Evaluate and save
best_model.to(DEVICE)
final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
print(f'Best PPL: {final_ppl}')
torch.save(best_model.state_dict(), 'bin/avsgd_lstm.pt')
