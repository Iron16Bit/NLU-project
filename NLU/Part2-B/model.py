import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BertIntentSlotModel(nn.Module):
    """
    Joint Intent Classification and Slot Filling using BERT
    Based on the approach from https://arxiv.org/abs/1902.10909
    """
    def __init__(self, bert_model_name, num_intents, num_slots, dropout_rate=0.1):
        super(BertIntentSlotModel, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Intent classification layer
        self.intent_classifier = nn.Linear(self.hidden_size, num_intents)
        
        # Slot filling layer
        self.slot_classifier = nn.Linear(self.hidden_size, num_slots)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, subword_indices=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention mask (1 for tokens, 0 for padding)
            token_type_ids: Tensor of segment ids (optional, not needed for single sequence)
            subword_indices: List of indices mapping subwords back to original tokens (for slot filling)
            
        Returns:
            intent_logits: Logits for intent classification
            slot_logits: Logits for slot filling (aligned with original tokens)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get the hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Intent classification from [CLS] token (pooled output)
        intent_logits = self.intent_classifier(pooled_output)
        
        # Slot filling - handle subword tokens
        batch_size = sequence_output.size(0)
        slot_logits = self.slot_classifier(sequence_output)  # [batch_size, seq_len, num_slots]
        
        # If we have subword mapping info, align subword representations to original tokens
        if subword_indices is not None:
            aligned_slot_logits = []
            for i in range(batch_size):
                # Get the first subword of each original token
                orig_token_indices = subword_indices[i]
                # Select the representations of these subwords
                aligned_logits = slot_logits[i, orig_token_indices, :]
                aligned_slot_logits.append(aligned_logits)
            
            # Pad sequences to same length
            aligned_slot_logits = pad_sequence(aligned_slot_logits, batch_first=True)
            # For computing the loss, we'll permute to [batch, classes, seq_len]
            aligned_slot_logits = aligned_slot_logits.permute(0, 2, 1)
            return intent_logits, aligned_slot_logits
        
        # For raw prediction without subword mapping (permute for CrossEntropyLoss)
        slot_logits = slot_logits.permute(0, 2, 1)  # [batch_size, num_slots, seq_len]
        return intent_logits, slot_logits


class JointBertDataset(Dataset):
    """
    Dataset for joint intent classification and slot filling with BERT
    Handles subword tokenization and alignment with original tokens
    """
    def __init__(self, texts, slot_labels, intent_labels, tokenizer, max_length=128):
        self.texts = texts  # List of utterances
        self.slot_labels = slot_labels  # List of lists containing slot labels for each token
        self.intent_labels = intent_labels  # List of intent labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        intent_label = self.intent_labels[idx]
        slot_labels = self.slot_labels[idx]
        
        # Tokenize the text into words
        words = text.split()
        
        # Lists to store the input features
        input_ids_list = []
        subword_indices = []  # To map subwords back to original tokens
        
        # Process each word and keep track of subwords
        word_index = 1  # Start at 1 to account for [CLS] token
        for i, word in enumerate(words):
            # Tokenize word into subwords
            subwords = self.tokenizer.tokenize(word)
            
            # Add subword tokens to input_ids
            for subword in subwords:
                input_ids_list.append(self.tokenizer.convert_tokens_to_ids(subword))
            
            # Record the index of the first subword of each original token
            subword_indices.append(word_index)
            
            # Update word_index to point to the next word's first subword
            word_index += len(subwords)
        
        # Add special tokens
        input_ids = [self.tokenizer.cls_token_id] + input_ids_list + [self.tokenizer.sep_token_id]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad or truncate sequences
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            # Adjust subword_indices if we truncated
            subword_indices = [idx for idx in subword_indices if idx < self.max_length - 1]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        intent_label = torch.tensor(intent_label)
        
        # Handle slot labels (we'll align them with subwords during training)
        slot_labels_tensor = torch.tensor(slot_labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'intent_label': intent_label,
            'slot_labels': slot_labels_tensor,
            'subword_indices': subword_indices
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences and subword indices
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    intent_labels = torch.stack([item['intent_label'] for item in batch])
    
    # For slot labels, we need to handle variable lengths
    slot_labels = [item['slot_labels'] for item in batch]
    max_len = max(len(labels) for labels in slot_labels)
    
    # Pad slot labels
    padded_slot_labels = []
    for labels in slot_labels:
        padding = torch.full((max_len - len(labels),), -100)  # -100 is ignored by CrossEntropyLoss
        padded_slot_labels.append(torch.cat((labels, padding)))
    
    slot_labels = torch.stack(padded_slot_labels)
    
    # Collect subword indices (these will be used during forward pass)
    subword_indices = [item['subword_indices'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'intent_labels': intent_labels,
        'slot_labels': slot_labels,
        'subword_indices': subword_indices
    }


def train_loop(dataloader, model, optimizer, scheduler, intent_criterion, slot_criterion, device):
    """
    Training loop for one epoch
    """
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intent_labels = batch['intent_labels'].to(device)
        slot_labels = batch['slot_labels'].to(device)
        subword_indices = batch['subword_indices']
        
        # Forward pass
        intent_logits, slot_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            subword_indices=subword_indices
        )
        
        # Calculate losses
        intent_loss = intent_criterion(intent_logits, intent_labels)
        
        # For slot loss, we need to reshape to match CrossEntropyLoss expectations
        slot_loss = slot_criterion(slot_logits, slot_labels)
        
        # Combine losses (can be weighted if needed)
        loss = intent_loss + slot_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def eval_loop(dataloader, model, intent_criterion, slot_criterion, device, lang):
    """
    Evaluation loop
    """
    model.eval()
    total_loss = 0
    intent_preds = []
    intent_labels = []
    slot_preds = []
    slot_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_intent_labels = batch['intent_labels'].to(device)
            batch_slot_labels = batch['slot_labels'].to(device)
            subword_indices = batch['subword_indices']
            
            # Forward pass
            intent_logits, slot_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                subword_indices=subword_indices
            )
            
            # Calculate losses
            intent_loss = intent_criterion(intent_logits, batch_intent_labels)
            slot_loss = slot_criterion(slot_logits, batch_slot_labels)
            loss = intent_loss + slot_loss
            
            total_loss += loss.item()
            
            # Get predictions
            intent_pred = torch.argmax(intent_logits, dim=1)
            
            # Transpose slot_logits back for prediction
            slot_logits = slot_logits.permute(0, 2, 1)  # [batch_size, seq_len, num_slots]
            slot_pred = torch.argmax(slot_logits, dim=2)
            
            # Collect predictions and labels for metrics calculation
            intent_preds.extend(intent_pred.cpu().numpy())
            intent_labels.extend(batch_intent_labels.cpu().numpy())
            
            # For slots, we need to handle padding
            for i in range(len(batch_slot_labels)):
                # Find where actual tokens end (not padding)
                actual_length = (batch_slot_labels[i] != -100).sum().item()
                if actual_length > 0:
                    slot_preds.append(slot_pred[i, :actual_length].cpu().numpy())
                    slot_labels.append(batch_slot_labels[i, :actual_length].cpu().numpy)