import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from model import LM_RNN, LM_LSTM_0, LM_LSTM
from functions import init_weights, train_loop, eval_loop
from utils import lang, train_loader, dev_loader, test_loader
from config import lr, emb_size, hidden_size, DEVICE, patience, n_epochs, clip, dropout

bin_dir = 'bin'
model_files = [f for f in os.listdir(bin_dir) if f.endswith('.pt')]
model_map = {
    'rnn.pt': LM_RNN,
    'lstm.pt': LM_LSTM_0,
    'dropout_lstm.pt': LM_LSTM,
    'dropout_lstm_adam.pt': LM_LSTM
}

print("Available models in bin/:")
for idx, fname in enumerate(model_files):
    print(f"{idx+1}. {fname}")

choice = int(input("Select a model to load (number): ")) - 1
model_fname = model_files[choice]
model_path = os.path.join(bin_dir, model_fname)

# Select the correct class
ModelClass = model_map[model_fname]
vocab_len = len(lang.word2id)

# Model instantiation based on class
if model_fname == 'rnn.pt':
    model = ModelClass(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
elif model_fname == 'lstm.pt':
    model = ModelClass(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
else:  # dropout_lstm.pt or dropout_lstm_adam.pt
    model = ModelClass(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"], emb_dropout=dropout, out_dropout=dropout).to(DEVICE)

model.load_state_dict(torch.load(model_path))
print(f"Loaded model weights from {model_path}")
print(f"Model architecture: {ModelClass.__name__}")

mode = input("Type 'eval' to evaluate or 'train' to train: ").strip().lower()

criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

if mode == 'eval':
    # Evaluation mode
    model.eval()
    val_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    print(f"Validation PPL: {val_ppl}")

elif mode == 'train':
    # Training mode - only train the selected model
    print(f"\nTraining {model_fname} ...")
    
    # Reset the model weights
    if model_fname == 'rnn.pt':
        model = ModelClass(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    elif model_fname == 'lstm.pt':
        model = ModelClass(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    else:
        model = ModelClass(emb_size, hidden_size, vocab_len, pad_index=lang.word2id["<pad>"], emb_dropout=dropout, out_dropout=dropout).to(DEVICE)
    
    model.apply(init_weights)
    
    # Select optimizer based on model name
    if model_fname == "dropout_lstm_adam.pt":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    best_local_ppl = float('inf')
    local_patience = patience
    best_local_model = None
    checkpoint_path = os.path.join(bin_dir, f'{model_fname}_best_checkpoint.pt')

    for epoch in range(n_epochs):
        # Train one epoch
        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        
        # Evaluate
        model.eval()
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
        model.train()
        
        print(f"Epoch {epoch+1}: lr={lr}, train_loss={train_loss:.4f}, val_PPL={ppl_dev:.2f}")

        if ppl_dev < best_local_ppl:
            best_local_ppl = ppl_dev
            best_local_model = copy.deepcopy(model)
            local_patience = patience
            
            # Save temporary checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'validation_ppl': ppl_dev,
                'epoch': epoch
            }, checkpoint_path)
            
            print(f"New best validation PPL: {ppl_dev:.2f}")
        else:
            local_patience -= 1
            print(f"No improvement. Patience: {local_patience}")

        if local_patience <= 0:
            print("Early stopping triggered")
            break

    # Evaluate final best model on validation set
    best_local_model.to(DEVICE)
    best_local_model.eval()
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_local_model)
    print(f'Best validation PPL for {model_fname}: {final_ppl:.2f}')
    
    # Save just the weights to the standard location
    torch.save(best_local_model.state_dict(), os.path.join(bin_dir, model_fname))
    print(f'Model saved to {os.path.join(bin_dir, model_fname)}')
    
    # Delete checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Deleted checkpoint file: {checkpoint_path}")
else:
    print("Invalid mode. Please type 'eval' or 'train'.")