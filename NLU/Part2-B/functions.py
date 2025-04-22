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
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    #origin_utt, _ = merge(new_item['original_utterance_ids'])
    
    #origin_utt.to(device)
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
   #new_item['original_utterances'] = new_item['original_utterance']
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item



def build_dataloaders(train_raw, val_raw, test_raw, lang, tokenizer):

    train_dataset = utils.IntentsAndSlots(train_raw, lang, tokenizer=tokenizer, myType='train')
    val_dataset = utils.IntentsAndSlots(val_raw, lang, tokenizer=tokenizer, myType='val')
    test_dataset = utils.IntentsAndSlots(test_raw, lang, tokenizer=tokenizer, myType='test')

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def train(model, data, optimizer, clip, criterion_slots, criterion_intents, device):
    """
    Enhanced training function for a BERT-based model with better monitoring
    """
    print("Starting batch training...")
    model.train()
    total_loss = []
    
    print(f"Number of batches: {len(data)}")
    
    for i, batch in enumerate(data):
        if i == 0:
            print(f"Processing first batch, keys: {batch.keys()}")
        
        # Move all tensors in batch to device
        input_ids = batch['utterances'].to(device)
        
        # Handle attention mask
        if 'attention_mask' in batch:
            if isinstance(batch['attention_mask'], list):
                try:
                    attention_mask = torch.stack(batch['attention_mask']).to(device)
                except:
                    attention_mask = torch.ones_like(input_ids).to(device)
            else:
                attention_mask = batch['attention_mask'].to(device)
        else:
            attention_mask = torch.ones_like(input_ids).to(device)
        
        # Get y_slots and intents
        slots = batch['y_slots'].to(device) if 'y_slots' in batch else None
        intents = batch['intents'].to(device) if 'intents' in batch else None
        
        # Clear gradients
        optimizer.zero_grad()
        
        try:
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
            
            # Fix the mismatch - intent_logits and slot_logits are swapped
            # The outputs appear to be flipped compared to what the validation expects
            
            # For slots - check if we have the expected shape [batch_size, seq_len, num_slots]
            if len(slot_logits.shape) == 3:
                # Already in the correct format
                slots_logits = slot_logits
            elif len(intent_logits.shape) == 3:
                # The outputs are swapped
                slots_logits = intent_logits
                intent_logits = slot_logits
            else:
                # Something is wrong with the model outputs
                raise ValueError(f"Expected either intent_logits or slot_logits to have shape [batch_size, seq_len, num_slots], got {intent_logits.shape} and {slot_logits.shape}")
            
            # For intent loss - if we don't have the expected shape [batch_size, num_intents]
            if len(intent_logits.shape) != 2:
                raise ValueError(f"Expected intent_logits to have shape [batch_size, num_intents], got {intent_logits.shape}")
            
            # Calculate losses with focal loss component for slots
            if len(slots_logits.shape) == 3:
                batch_size, seq_len, num_slots = slots_logits.shape
                
                # Apply label smoothing for slot predictions
                slots_flat = slots.view(-1)
                valid_indices = slots_flat != 0  # Ignore PAD_TOKEN (0)
                
                # Standard cross-entropy for slots
                loss_slots = criterion_slots(
                    slots_logits.view(-1, num_slots),  # [batch_size*seq_len, num_slots]
                    slots_flat                         # [batch_size*seq_len]
                )
                
                # Add focal loss component for rare slots
                if valid_indices.sum() > 0:
                    # Get probabilities for active predictions
                    probs = F.softmax(slots_logits.view(-1, num_slots), dim=1)
                    
                    # Get target probabilities (one-hot)
                    target_one_hot = torch.zeros_like(probs).scatter_(
                        1, slots_flat.unsqueeze(1), 1.0
                    )
                    
                    # Calculate focal term: (1 - p_t)^gamma
            
            loss_intent = criterion_intents(intent_logits, intents)
            
            # Combined loss
            loss = (3.0 * loss_slots) + loss_intent
            
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
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add more debugging info
            if 'slots_logits' in locals() and 'slots' in locals():
                print(f"slots_logits shape: {slots_logits.shape}")
                print(f"slots shape: {slots.shape}")
                print(f"After reshape:")
                print(f"slots_logits.view(-1, slots_logits.shape[-1]) shape: {slots_logits.view(-1, slots_logits.shape[-1]).shape}")
                print(f"slots.view(-1) shape: {slots.view(-1).shape}")
            continue
    
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
        
        for sample in data:
            input_ids = sample['utterances'].to(device)
            
            # Handle attention mask
            if 'attention_mask' in sample:
                if isinstance(sample['attention_mask'], list):
                    try:
                        attention_mask = torch.stack(sample['attention_mask']).to(device)
                    except:
                        attention_mask = torch.ones_like(input_ids).to(device)
                else:
                    attention_mask = sample['attention_mask'].to(device)
            else:
                attention_mask = torch.ones_like(input_ids).to(device)
            
            # Get labels
            intents = sample['intents'].to(device)
            slots = sample['y_slots'].to(device)

            # Forward pass
            intent_logits, slot_logits = model(token_ids=input_ids, attention_mask=attention_mask)
            
            # Swap outputs if needed based on shapes
            if len(slot_logits.shape) == 3 and len(intent_logits.shape) == 2:
                # Already correct format
                pass
            elif len(intent_logits.shape) == 3 and len(slot_logits.shape) == 2:
                # Swap them
                temp = slot_logits
                slot_logits = intent_logits
                intent_logits = temp
            
            # Calculate losses with proper shapes
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

            # Process slot predictions - this is the critical part
            predicted_slots = torch.argmax(slot_logits, dim=2)  # [batch_size, seq_len]
            
            # Process each sequence in the batch
            for idx, (pred_seq, true_seq) in enumerate(zip(predicted_slots, slots)):
                length = sample['slots_len'].tolist()[idx]
                
                # Skip padding - only consider actual tokens
                # Start from index 1 to skip [CLS] token
                pred_seq = pred_seq[1:length].tolist()
                true_seq = true_seq[1:length].tolist()
                
                # Get the original words for this utterance
                # Fix: properly handle the utterance field which could be a tensor or string
                if isinstance(sample['utterance'], list):
                    if isinstance(sample['utterance'][idx], torch.Tensor):
                        utterance = tokenizer.decode(sample['utterance'][idx]).split()
                    else:
                        utterance = sample['utterance'][idx].split()
                else:
                    # If utterance is not directly accessible, try to decode from input_ids
                    utterance = tokenizer.decode(input_ids[idx]).split()
                
                # Ensure we have the right sequence length 
                # Sometimes the tokenizer adds special tokens or splits words differently
                min_len = min(len(utterance), len(pred_seq), len(true_seq))
                
                # Map to slot labels
                try:
                    ref_seq = [(utterance[i] if i < len(utterance) else "<PAD>", 
                               lang.id2slot[true_seq[i]]) for i in range(min_len)]
                    hyp_seq = [(utterance[i] if i < len(utterance) else "<PAD>", 
                               lang.id2slot[pred_seq[i]]) for i in range(min_len)]
                    
                    ref_slots.append(ref_seq)
                    hyp_slots.append(hyp_seq)
                except Exception as e:
                    print(f"Error processing sequence {idx}: {e}")
                    # Print debugging info
                    print(f"min_len: {min_len}, utterance len: {len(utterance)}, pred_seq len: {len(pred_seq)}, true_seq len: {len(true_seq)}")
                    print(f"utterance: {utterance[:10]}...")
        
    # Evaluate slot predictions
    try:
        results = evaluate(ref_slots, hyp_slots)
        # Print some examples for debugging
        if len(ref_slots) > 0:
            print("\nSample slot predictions:")
            print("Ref:", ref_slots[0][:10])
            print("Hyp:", hyp_slots[0][:10])
    except Exception as e:
        print(f"Error in slot evaluation: {e}")
        # Sometimes the model predicts a class that is not in REF
        ref_s = set([x[1] for seq in ref_slots for x in seq])
        hyp_s = set([x[1] for seq in hyp_slots for x in seq])
        print(f"Missing slot labels: {hyp_s.difference(ref_s)}")
        results = {"total": {"f": 0}}

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