import torch.nn as nn

def apply_variational_dropout(x, dropout_prob):
    # Ensure the tensor is not in evaluation mode and dropout is needed
    if not x.requires_grad or dropout_prob == 0:
        return x
    
    batch_size, seq_len, dim = x.size()
    
    # Generate the dropout mask using Bernoulli distribution
    mask = x.new_empty(batch_size, seq_len, dim).bernoulli_(1 - dropout_prob)
    
    # Scale the mask so the remaining values are appropriately scaled (inverse of dropout rate)
    mask = mask.div_(1 - dropout_prob)  # This keeps the expected value of x the same
    
    # Apply the mask to the input tensor
    return x * mask

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, dropout=0.1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size, bias=False)  # No bias for weight tying
        # Use the same weights for embedding and output layers
        self.output.weight = self.embedding.weight
        
        # Dropout probability
        self.dropout = dropout

    def forward(self, input_sequence):
        batch_size, _ = input_sequence.size()
        emb = self.embedding(input_sequence)
        
        # Apply variational dropout to embeddings (same mask for all timesteps)
        if self.training and self.dropout > 0:
            emb = apply_variational_dropout(emb, self.dropout)
            
        lstm_out, _ = self.lstm(emb)
        
        # Apply variational dropout to LSTM output (same mask for all timesteps)
        if self.training and self.dropout > 0:
            lstm_out = apply_variational_dropout(lstm_out, self.dropout)
            
        output = self.output(lstm_out).permute(0, 2, 1)
        return output