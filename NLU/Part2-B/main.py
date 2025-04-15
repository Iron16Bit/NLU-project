import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Import our model and dataset classes
from model import (
    BertIntentSlotModel, 
    JointBertDataset, 
    collate_fn, 
    train_loop, 
    eval_loop
)

# Import from existing code
from utils import lang, train_loader, dev_loader, test_loader
from config import device, clip, lr, n_epochs, runs

# Global variables
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Model configurations
bert_model_name = "bert-base-uncased"  # You can change to "bert-large-uncased" if you have enough resources
num_intents = len(lang.intent2id)
num_slots = len(lang.slot2id)
PAD_TOKEN = -100  # Use -100 for CrossEntropyLoss to ignore padding

# Training configurations
warmup_steps = 0
weight_decay = 0.01
dropout_rates = [0.1, 0.2, 0.3]  # Different dropout rates to experiment with

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Function to preprocess the data and create dataset objects
def preprocess_data(texts, slot_labels, intent_labels, tokenizer, max_length=128):
    dataset = JointBertDataset(
        texts=texts,
        slot_labels=slot_labels,
        intent_labels=intent_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    return dataset

# Extract raw data from the existing DataLoader objects
def extract_data_from_loader(loader):
    texts, slot_labels, intent_labels = [], [], []
    
    for batch in loader.dataset:
        # Assuming each batch has a text, intent_label, and slot_labels
        texts.append(batch[0])  # Assuming text is the first item
        intent_labels.append(batch[1])  # Assuming intent label is the second item
        slot_labels.append(batch[2])  # Assuming slot labels are the third item
    
    return texts, slot_labels, intent_labels

# Extract data for creating our BERT datasets
train_texts, train_slot_labels, train_intent_labels = extract_data_from_loader(train_loader)
dev_texts, dev_slot_labels, dev_intent_labels = extract_data_from_loader(dev_loader)
test_texts, test_slot_labels, test_intent_labels = extract_data_from_loader(test_loader)

# Create datasets
train_dataset = preprocess_data(train_texts, train_slot_labels, train_intent_labels, tokenizer)
dev_dataset = preprocess_data(dev_texts, dev_slot_labels, dev_intent_labels, tokenizer)
test_dataset = preprocess_data(test_texts, test_slot_labels, test_intent_labels, tokenizer)

# Create DataLoaders
bert_train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_fn
)

bert_dev_loader = DataLoader(
    dev_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=collate_fn
)

bert_test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=collate_fn
)

# Loss functions
intent_criterion = nn.CrossEntropyLoss()
slot_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

# Function to initialize weights
def init_bert_weights(model):
    """Initialize weights for non-BERT layers"""
    for name, param in model.named_parameters():
        if "bert" not in name:  # Only initialize non-BERT weights
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param)
    return model

# Experiment with different dropout rates
slot_f1s_all = []
intent_acc_all = []

for dropout_rate in dropout_rates:
    print(f"\n=== Training with Dropout: {dropout_rate} ===")
    
    slot_f1s = []
    intent_acc = []
    
    for run in tqdm(range(0, runs)):
        # Initialize model
        model = BertIntentSlotModel(
            bert_model_name=bert_model_name,
            num_intents=num_intents,
            num_slots=num_slots,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Apply weight initialization for non-BERT layers
        model = init_bert_weights(model)
        
        # Group parameters for different learning rates
        # BERT layers usually need a smaller learning rate than the task-specific layers
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param_optimizer = list(model.bert.named_parameters())
        classifier_param_optimizer = list(model.intent_classifier.named_parameters()) + \
                                     list(model.slot_classifier.named_parameters())
        
        optimizer_grouped_parameters = [
            # BERT parameters with weight decay
            {
                'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': lr / 10  # Lower learning rate for BERT
            },
            # BERT parameters without weight decay
            {
                'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': lr / 10
            },
            # Task-specific parameters with weight decay
            {
                'params': [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': lr
            },
            # Task-specific parameters without weight decay
            {
                'params': [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': lr
            }
        ]
        
        # Initialize optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        
        # Calculate total steps for learning rate scheduler
        total_steps = len(bert_train_loader) * n_epochs
        
        # Create scheduler with linear warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        patience = 3
        best_f1 = 0
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        
        for epoch in range(1, n_epochs):
            # Train
            loss = train_loop(
                dataloader=bert_train_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                intent_criterion=intent_criterion,
                slot_criterion=slot_criterion,
                device=device
            )
            
            if epoch % 5 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(loss)
                
                # Evaluate on dev set
                results_dev, intent_res, loss_dev = eval_loop(
                    dataloader=bert_dev_loader,
                    model=model,
                    intent_criterion=intent_criterion,
                    slot_criterion=slot_criterion,
                    device=device,
                    lang=lang
                )
                
                losses_dev.append(loss_dev)
                
                # Check if we should stop early
                f1 = results_dev['total']['f']
                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                
                if patience <= 0:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Evaluate on test set
        results_test, intent_test, _ = eval_loop(
            dataloader=bert_test_loader,
            model=model,
            intent_criterion=intent_criterion,
            slot_criterion=slot_criterion,
            device=device,
            lang=lang
        )
        
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
    
    # After all runs for this dropout value
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    slot_f1s_all.append(slot_f1s)
    intent_acc_all.append(intent_acc)
    
    print(f'Dropout {dropout_rate} -> Slot F1: {round(slot_f1s.mean(), 3)} ± {round(slot_f1s.std(), 3)}')
    print(f'Dropout {dropout_rate} -> Intent Acc: {round(intent_acc.mean(), 3)} ± {round(intent_acc.std(), 3)}')

print("\n=== Summary of All Dropout Settings ===")
for i, dropout_rate in enumerate(dropout_rates):
    print(f'Dropout {dropout_rate} -> Slot F1: {round(slot_f1s_all[i].mean(), 3)} ± {round(slot_f1s_all[i].std(), 3)}, '
          f'Intent Acc: {round(intent_acc_all[i].mean(), 3)} ± {round(intent_acc_all[i].std(), 3)}')

# Find best dropout rate based on F1 score
best_dropout_idx = np.argmax([s.mean() for s in slot_f1s_all])
best_dropout = dropout_rates[best_dropout_idx]
print(f"\nBest dropout rate: {best_dropout}")

# Plot results for the best dropout rate
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot(slot_f1s_all)
plt.xticks(range(1, len(dropout_rates) + 1), dropout_rates)
plt.title('Slot F1 Scores by Dropout Rate')
plt.xlabel('Dropout Rate')
plt.ylabel('F1 Score')

plt.subplot(1, 2, 2)
plt.boxplot(intent_acc_all)
plt.xticks(range(1, len(dropout_rates) + 1), dropout_rates)
plt.title('Intent Accuracy by Dropout Rate')
plt.xlabel('Dropout Rate')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
