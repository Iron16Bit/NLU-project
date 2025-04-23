# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

#what should i put in model.py then???
import matplotlib.pyplot as plt
import torch
import math

from functools import partial
from torch import LongTensor
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import classification_report

from config import PAD_TOKEN, device
import utils
from conll import evaluate

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.pad_token = 0
        self.word2id = self.words2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.label2id(slots)
        self.intent2id = self.label2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def words2id(self, elements, cutoff=None, unk=True):
        from collections import Counter
        vocab = {'pad': self.pad_token}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def label2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

def collate_fn(data, pad_token=0, device='cuda:0'):
    """
    Modified collate function to handle both tensor and string types for utterances
    """
    # Sort data by seq lengths (needed for pack_padded_sequence later)
    data.sort(key=lambda x: len(x['utterances']), reverse=True)
    
    # Create a dictionary to hold batched values
    batch_data = {}
    
    # Handle utterance IDs (already tensors)
    if 'utterances' in data[0]:
        utterances = [d['utterances'] for d in data]
        batch_data['utterances'] = torch.stack(utterances).to(device)
    
    # Handle attention_mask
    if 'attention_mask' in data[0]:
        attention_masks = [d['attention_mask'] for d in data]
        batch_data['attention_mask'] = torch.stack(attention_masks).to(device)
    
    # Handle slots (already tensors)
    if 'slots' in data[0]:
        slots = [d['slots'] for d in data]
        batch_data['y_slots'] = torch.stack(slots).to(device)
        # Calculate actual sequence lengths for slots
        lengths = []
        for s in slots:
            # Find the last non-padding token
            mask = (s != pad_token).int()
            length = mask.sum().item()
            lengths.append(length if length > 0 else 1)  # Minimum length of 1
        batch_data['slots_len'] = torch.LongTensor(lengths).to(device)
    
    # Handle intent (already a tensor)
    if 'intent' in data[0]:
        intents = [d['intent'] for d in data]
        batch_data['intents'] = torch.tensor(intents, dtype=torch.long).to(device)
    
    # Keep the original utterance strings for reference
    if 'utterance' in data[0]:
        batch_data['utterance'] = [d['utterance'] for d in data]
    
    return batch_data

def build_dataloaders(train_raw, val_raw, test_raw, lang, tokenizer):
    """Updated dataloader builder that correctly passes the collate function"""
    train_dataset = utils.IntentsAndSlots(train_raw, lang, tokenizer=tokenizer, myType='train')
    val_dataset = utils.IntentsAndSlots(val_raw, lang, tokenizer=tokenizer, myType='val')
    test_dataset = utils.IntentsAndSlots(test_raw, lang, tokenizer=tokenizer, myType='test')

    # Specify our custom collate function with default parameters
    collate = lambda x: collate_fn(x, pad_token=PAD_TOKEN, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate)

    return train_loader, val_loader, test_loader

def train(model, data, optimizer, clip, criterion_slots, criterion_intents, device):
    model.train()
    print("Starting batch training...")
    total_loss = []
    
    print(f"Number of batches: {len(data)}")
    
    for i, batch in enumerate(data):
        if i == 0:
            print(f"Processing first batch, keys: {batch.keys()}")
        
        # Get inputs and labels
        input_ids = batch['utterances'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
        slots = batch['y_slots'].to(device)
        intents = batch['intents'].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        intent_logits, slot_logits = model(
            token_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if i == 0:
            print("Forward pass successful")
            print(f"Intent logits shape: {intent_logits.shape}")
            print(f"Slots logits shape: {slot_logits.shape}")
            print(f"Slots shape: {slots.shape}")
            print(f"Intents shape: {intents.shape}")
        
        # Compute losses
        batch_size, seq_len, num_slots = slot_logits.shape
        
        # Reshape for loss calculation
        slot_logits_view = slot_logits.view(-1, num_slots)
        slots_view = slots.view(-1)
        
        # Calculate losses
        loss_slots = criterion_slots(slot_logits_view, slots_view)
        loss_intent = criterion_intents(intent_logits, intents)
        
        # Combined loss - weight slot loss higher since it's harder to learn
        loss = (2.0 * loss_slots) + loss_intent
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        # Store loss
        total_loss.append(loss.item())
        
        if i == 0:
            print(f"First batch processed successfully, loss: {loss.item()}")
        
        if i % 10 == 0 and i > 0:
            print(f"Completed {i}/{len(data)} batches")
    
    print("Training epoch completed")
    return total_loss

# basically same as training, without the backward of the loss and the addition of the evaluation of the performances
def validation(model, data, lang, criterion_slots, criterion_intents, tokenizer, device='cuda:0'):
    model.eval()

    # validation, don't compute grads
    with torch.no_grad():
        total_loss = 0
        ref_intents = []
        hyp_intents = []
        
        ref_slots = []
        hyp_slots = []
        
        for batch in data:
            input_ids = batch['utterances'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            slots = batch['y_slots'].to(device)
            intents = batch['intents'].to(device)

            # Forward pass
            intent_logits, slot_logits = model(token_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate losses
            batch_size, seq_len, num_slots = slot_logits.shape
            
            loss_slot = criterion_slots(
                slot_logits.view(-1, num_slots), 
                slots.view(-1)
            )
            
            loss_intent = criterion_intents(intent_logits, intents)
            loss = loss_intent + loss_slot
            total_loss += loss.item()

            # Process intent predictions
            predicted_intents = [lang.id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()]
            real_intents = [lang.id2intent[x] for x in intents.tolist()]
            ref_intents.extend(real_intents)
            hyp_intents.extend(predicted_intents)

            # Process slot predictions
            predicted_slots = torch.argmax(slot_logits, dim=2)  # [batch_size, seq_len]
            
            # Process each sequence in the batch
            for idx, (pred_seq, true_seq, length) in enumerate(zip(predicted_slots, slots, batch['slots_len'])):
                length = length.item()
                
                # Get the sequence up to its actual length, skipping [CLS]
                pred_seq = pred_seq[1:length].tolist()
                true_seq = true_seq[1:length].tolist()
                
                # Get original utterance for this sample
                original_utterance = batch['utterance'][idx].split()
                
                # Ensure we have the right sequence length
                min_len = min(len(original_utterance), len(pred_seq), len(true_seq))
                
                # Map sequences to slot labels
                ref_seq = [(original_utterance[i] if i < len(original_utterance) else "<PAD>", 
                           lang.id2slot[true_seq[i]]) for i in range(min_len)]
                hyp_seq = [(original_utterance[i] if i < len(original_utterance) else "<PAD>", 
                           lang.id2slot[pred_seq[i]]) for i in range(min_len)]
                
                ref_slots.append(ref_seq)
                hyp_slots.append(hyp_seq)
        
    # Evaluate slot predictions
    results = evaluate(ref_slots, hyp_slots)
    
    # Print some examples for debugging
    if len(ref_slots) > 0:
        print("\nSample slot predictions:")
        print("Ref:", ref_slots[0][:10])
        print("Hyp:", hyp_slots[0][:10])

    # Compute intent classification metrics
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    
    return results, report_intent, total_loss/len(data)

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def plot_results(data, epochs, label):
    epochs_list = range(1,epochs+1)

    plt.figure(figsize=(10,5))
    plt.plot(epochs_list, data, label=label)
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.title(label + ' evolution')
    plt.legend()

    plt.savefig(label+'.png')


def get_dataset_informations(train_raw_data, val_raw_data, test_raw_data):
    # want to have directly the intents of the test set too, just like we did for validation and training
    intent_test = [x['intent'] for x in test_raw_data]

    # list of all words in the train set (list as we want to compute frequency of each word too)
    words = sum([x['utterance'].split() for x in train_raw_data], [])

    # list of dictionaries, [{'utterance': 'x', 'slots':'x', 'intent':'airfare'}]
    corpus = train_raw_data + val_raw_data + test_raw_data

    # set slots eg: {'I-cost_relative, 'B-arrive_time.time', 'B-return_date.day_name', ...}
    slots = set(sum([line['slots'].split() for line in corpus],[]))

    # set of all the intents in the corpus
    total_intents = set([line['intent'] for line in corpus])

    return intent_test, words, corpus, slots, total_intents