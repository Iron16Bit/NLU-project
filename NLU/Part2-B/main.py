# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import StandardBERT

from tqdm import tqdm
from transformers import BertTokenizerFast
import copy
import math
import torch.optim as optim
import numpy as np
import torch.nn as nn
import os

from config import n_epochs, dropout, lr, PAD_TOKEN, clip

if __name__ == "__main__":
    patience_fixed = 5
    current_patience = patience_fixed

    losses_train = []
    losses_val = []
    sampled_epochs = []

    best_model = None
    pbar = tqdm(range(1, n_epochs + 1))

    GPU = "cuda:0" if torch.cuda.is_available() else "cpu"
    CPU = 'cpu'

    # Set a seed for reproducibility of experiments
    torch.manual_seed(32)
    exp_name = "standard_bert"

    # Use the local ATIS dataset
    print("Loading ATIS dataset from local directory...")
    atis_data = load_from_local_atis()
    
    train_raw_data = atis_data['train']
    test_raw_data = atis_data['test']
    
    print('Train samples:', len(train_raw_data))
    print('Test samples:', len(test_raw_data))

    # Build the validation set
    train_raw_data, intent_train, val_raw_data, intent_val = generate_validation_set(
        training_set_raw=train_raw_data, 
        percentage=0.1
    )

    intent_test, words, corpus, slots, total_intents = get_dataset_informations(
        train_raw_data=train_raw_data, 
        val_raw_data=val_raw_data, 
        test_raw_data=test_raw_data
    )

    lang = Lang(words, total_intents, slots, cutoff=0)
    slots.add('pad')

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    model_name = 'bert-base-uncased'
    max_token_len = 50

    # Use the standard BERT model instead of the modified one
    print(f"Creating standard BERT model with {len(total_intents)} intents and {len(slots)} slots")
    model = StandardBERT(num_intents=len(total_intents), num_slots=len(slots)).to(GPU)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_raw=train_raw_data, 
        val_raw=val_raw_data, 
        test_raw=test_raw_data, 
        lang=lang,
        tokenizer=tokenizer
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    losses_train = []
    losses_val = []
    sampled_epochs = []
    best_f1 = 0
    best_model = model

    print("Starting training...")
    for epoch in pbar:
        loss = train(model=model, data=train_loader, optimizer=optimizer, clip=clip, 
                     criterion_slots=criterion_slots, criterion_intents=criterion_intents, device=GPU)
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            results_val, intent_res, loss_val = validation(model=model, data=val_loader, lang=lang, 
                                                       criterion_slots=criterion_slots, criterion_intents=criterion_intents, 
                                                       tokenizer=tokenizer, device=GPU)
            losses_val.append(np.asarray(loss_val).mean())
        
            f1 = results_val['total']['f']

            pbar.set_description(f"f1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                current_patience = patience_fixed
                best_model = copy.deepcopy(model).to(CPU)
            else:
                current_patience -= 1

            if current_patience <= 0:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print("\nEvaluating on test set...")
    best_model = best_model.to(GPU)
    results_test, intent_test, _ = validation(model=best_model, data=test_loader, lang=lang, 
                                           criterion_slots=criterion_slots, criterion_intents=criterion_intents, 
                                           tokenizer=tokenizer, device=GPU)
    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Save the best model
    torch.save(best_model.state_dict(), f"./models/{exp_name}.pth")
    print(f"Best model saved to ./models/{exp_name}.pth")