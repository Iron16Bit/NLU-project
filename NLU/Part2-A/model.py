import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Part 2-A.1 AIS with no bidirectionality
class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)    
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    
# Part 2-A.1 bidirectional AIS
# bidirectionality = the input sequence is read in both directions, then information are combined
# Why try to disambiguate by also reading the "future" of the word
class ModelIAS_bi(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_bi, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # bidirectionality = True
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        # doubled sizes due to biridectionality    
        self.slot_out = nn.Linear(2 * hid_size, out_slot)
        self.intent_out = nn.Linear(2 * hid_size, out_int)

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        # The last hidden layer is where the forward and backward readings meet, combine them
        last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        slots = slots.permute(0,2,1)
        return slots, intent
    
# Part 2-A.2 apply dropout layers
# Dropouts on embedding and hidden states improves generalization: https://arxiv.org/abs/1512.05287
# Dropouts should be applied on non-recurrent connections (embeddings, outputs) and not recurrent (hidden-to-hidden)
    # https://arxiv.org/abs/1409.2329
class ModelIAS_drop(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout=0.1):
        super(ModelIAS_drop, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        self.slot_out = nn.Linear(2 * hid_size, out_slot)
        self.intent_out = nn.Linear(2 * hid_size, out_int)

        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

    def forward(self, utterance, seq_lengths):
        utt_emb = self.dropout(self.embedding(utterance))  # Dropout after embedding
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)  # Dropout after LSTM outputs
        last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        last_hidden = self.dropout(last_hidden)  # Dropout before intent classification
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(0, 2, 1)
        return slots, intent