import torch
import torch.nn as nn
import torch.nn.functional as F

# Part 1-B.1
# Original LSTM
class LM_LSTM_0(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_0, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    
# Part 1-B.1
# LSTM with Weight Tying
class LM_LSTM_wt(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_wt, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size, bias=False) 
        # Weight Tying - Use the same weights
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output

# ----------------------------------------------------------------------------------------------------------------

# Part 1-B.2
# Variational Dropout
def apply_variational_dropout(x, dropout_prob):
    # Get tensor dimensions
    if x.dim() == 3:
        batch_size, seq_len, hidden_size = x.size()
        # Create mask (batch_size, 1, hidden_size) - same mask for all time steps
        mask = x.new_empty(batch_size, 1, hidden_size).bernoulli_(1 - dropout_prob)
    else:
        batch_size, hidden_size = x.size()
        # Create mask (batch_size, hidden_size)
        mask = x.new_empty(batch_size, hidden_size).bernoulli_(1 - dropout_prob)
    
    # Scale the mask
    mask = mask.div(1 - dropout_prob)
    
    # Apply the mask
    return x * mask

class VariationalDropoutLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, batch_first=True):
        super(VariationalDropoutLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Use PyTorch's built-in LSTM with dropout=0 as basis
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=batch_first,
            dropout=0.0  # No dropout in the LSTM itself
        )
        
    def forward(self, x, hidden=None):
        # Generate new hidden states if not provided
        if hidden is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            hidden = (h, c)
        else:
            h, c = hidden
            
        if self.training and self.dropout > 0:
            # Apply dropout to the input embedding
            x = apply_variational_dropout(x, self.dropout)
            
            # Create the specified number of layers and apply the dropout mask to them
            h_masks = []
            for layer in range(self.num_layers):
                # Create mask for hidden state (batch_size, hidden_size)
                batch_size = h.size(1)
                h_mask = h.new_empty(batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                h_mask = h_mask.div(1 - self.dropout)
                h_masks.append(h_mask)
                
                # Apply mask to this layer's hidden state
                h[layer] = h[layer] * h_masks[layer]

            # Replace the hidden state in the hidden tuple
            hidden = (h, c)
            
        output, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply the same dropout masks from before to the output hidden states
        if self.training and self.dropout > 0:
            last_h_mask = h_masks[-1].unsqueeze(1)  # Add sequence dimension
            if self.batch_first:
                # (batch_size, seq_len, hidden_size)
                output = output * last_h_mask
            else:
                # (seq_len, batch_size, hidden_size)
                output = output * last_h_mask.transpose(0, 1)
            
            # Apply masks to final hidden states
            for layer in range(self.num_layers):
                h_n[layer] = h_n[layer] * h_masks[layer]
                
        return output, (h_n, c_n)

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, dropout=0.1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.lstm = VariationalDropoutLSTM(
            emb_size, hidden_size, n_layers, 
            dropout=dropout, batch_first=True
        )
        
        self.pad_token = pad_index
        
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size, bias=False)
        
        # Use the same weights for embedding and output layers
        self.output.weight = self.embedding.weight
        
        # Dropout probability
        self.dropout = dropout
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        
        lstm_out, _ = self.lstm(emb)
        
        # Apply dropout to final output if needed
        if self.training and self.dropout > 0:
            lstm_out = apply_variational_dropout(lstm_out, self.dropout)
            
        output = self.output(lstm_out).permute(0, 2, 1)
        
        return output