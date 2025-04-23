# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import StandardBERT

from tqdm import tqdm
from transformers import BertTokenizerFast
import copy
import torch.optim as optim
import numpy as np
import torch.nn as nn
import os
from collections import Counter

from config import n_epochs, dropout, lr, PAD_TOKEN, clip, device, runs

if __name__ == "__main__":
    patience_fixed = 5
    
    # Create a dictionary to store results from all runs
    all_results = {
        'slot_f1': [],
        'intent_accuracy': []
    }
    
    print(f"Running {runs} experiments with {n_epochs} epochs each")
    
    for run in range(runs):
        print(f"\n=== Run {run+1}/{runs} ===")
        
        current_patience = patience_fixed
        losses_train = []
        losses_val = []
        sampled_epochs = []
        
        # Set a seed for reproducibility - different for each run
        torch.manual_seed(32 + run)
        exp_name = f"standard_bert_run_{run+1}"
        
        # Use the local ATIS dataset
        print("Loading ATIS dataset from local directory...")
        atis_data = load_from_local_atis()
        
        if 'train' in atis_data and 'test' in atis_data and 'dev' in atis_data:
            train_raw_data = atis_data['train']
            test_raw_data = atis_data['test']
            val_raw_data = atis_data['dev']
            
            print('Train samples:', len(train_raw_data))
            print('Validation samples:', len(val_raw_data))
            print('Test samples:', len(test_raw_data))
            
            # Skip the generate_validation_set call since we already have a validation set
            intent_train = [x['intent'] for x in train_raw_data]
            intent_val = [x['intent'] for x in val_raw_data]
        else:
            print("Failed to load ATIS dataset")
            exit(1)

        intent_test, words, corpus, slots, total_intents = get_dataset_informations(
            train_raw_data=train_raw_data, 
            val_raw_data=val_raw_data, 
            test_raw_data=test_raw_data
        )

        lang = Lang(words, total_intents, slots, cutoff=0)
        slots.add('pad')

        # Calculate class weights for slot labels to handle class imbalance
        slot_counts = Counter()
        for sample in train_raw_data:
            slot_counts.update(sample['slots'].split())

        # Create weights tensor: inverse-frequency weighting
        weights = torch.ones(len(slots), device=device)
        for slot_name, slot_id in lang.slot2id.items():
            if slot_name in slot_counts:
                # Higher weight (5x) for rare slots, less for common ones
                if slot_name != 'O' and slot_name != 'pad':
                    # Non-O tags get higher weights
                    weights[slot_id] = 5.0 * len(slot_counts) / slot_counts[slot_name]
                elif slot_name == 'O':
                    # 'O' tag gets lower weight
                    weights[slot_id] = 0.8 * len(slot_counts) / slot_counts[slot_name]

        # Normalize weights
        weights = weights / weights.mean()

        # Use weighted loss for slot classification
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # Simpler approach
        criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

        model_name = 'bert-base-uncased'
        max_token_len = 50

        # Creating model, tokenizer, criterion, optimizer for this run
        model = StandardBERT(num_intents=len(total_intents), num_slots=len(slots)).to(device)
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Is CUDA available: {torch.cuda.is_available()}")
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

        train_loader, val_loader, test_loader = build_dataloaders(
            train_raw=train_raw_data, 
            val_raw=val_raw_data, 
            test_raw=test_raw_data, 
            lang=lang,
            tokenizer=tokenizer
        )

        # After creating dataloaders
        print(f"Train loader size: {len(train_loader)}")
        print(f"Sample batch size: {next(iter(train_loader))['utterances'].shape}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # Add weight decay

        # Add a learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        best_f1 = 0
        best_model = model

        print("Starting training...")
        pbar = tqdm(range(1, n_epochs + 1), desc="Training epochs")
        for epoch in pbar:
            loss = train(model=model, data=train_loader, optimizer=optimizer, clip=clip, 
                            criterion_slots=criterion_slots, criterion_intents=criterion_intents, device=device)
            
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())

                results_val, intent_res, loss_val = validation(model=model, data=val_loader, lang=lang, 
                                                            criterion_slots=criterion_slots, criterion_intents=criterion_intents, 
                                                            tokenizer=tokenizer, device=device)
                losses_val.append(np.asarray(loss_val).mean())
            
                f1 = results_val['total']['f']

                pbar.set_description(f"f1: {f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    current_patience = patience_fixed
                    best_model = copy.deepcopy(model).to('cpu')
                else:
                    current_patience -= 1

                if current_patience <= 0:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Update learning rate based on F1 score
                scheduler.step(f1)
        
        print("\nEvaluating on test set...")
        best_model = best_model.to(device)
        results_test, intent_test, _ = validation(model=best_model, data=test_loader, lang=lang, 
                                                criterion_slots=criterion_slots, criterion_intents=criterion_intents, 
                                                tokenizer=tokenizer, device=device)
        
        slot_f1 = results_test['total']['f']
        intent_accuracy = intent_test['accuracy']
        
        print(f'Run {run+1} - Slot F1: {slot_f1}')
        print(f'Run {run+1} - Intent Accuracy: {intent_accuracy}')
        
        # Store results for this run
        all_results['slot_f1'].append(slot_f1)
        all_results['intent_accuracy'].append(intent_accuracy)
        
        # Create models directory if it doesn't exist
        os.makedirs("./models", exist_ok=True)
        
        # Save the best model for this run
        torch.save(best_model.state_dict(), f"./models/{exp_name}.pth")
        print(f"Best model for run {run+1} saved to ./models/{exp_name}.pth")
    
    # Print average results across all runs
    avg_slot_f1 = sum(all_results['slot_f1']) / runs
    avg_intent_accuracy = sum(all_results['intent_accuracy']) / runs
    
    print("\n=== Final Results ===")
    print(f"Average Slot F1 across {runs} runs: {avg_slot_f1:.4f}")
    print(f"Average Intent Accuracy across {runs} runs: {avg_intent_accuracy:.4f}")