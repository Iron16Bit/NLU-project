import torch.nn as nn

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
        self.output = nn.Linear(hidden_size, output_size)
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
    # Ensure the tensor is not in evaluation mode and dropout is needed
    if not x.requires_grad or dropout_prob == 0:
        return x
    
    batch_size = x.size(0)
    
    # For 3D tensors (batch_size, seq_len, dim)
    if x.dim() == 3:
        seq_len, dim = x.size(1), x.size(2)
        # Generate the dropout mask using Bernoulli distribution
        mask = x.new_empty(batch_size, 1, dim).bernoulli_(1 - dropout_prob)
    # For 2D tensors (batch_size, dim) - used for hidden states
    else:
        dim = x.size(1)
        # Generate the dropout mask using Bernoulli distribution
        mask = x.new_empty(batch_size, dim).bernoulli_(1 - dropout_prob)
    
    # Scale the mask so the remaining values are appropriately scaled
    mask = mask.div_(1 - dropout_prob)  # Keep the expected value of x the same
    
    # Apply the mask to the input tensor
    return x * mask

class VariationalDropoutLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(VariationalDropoutLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # LSTM cell parameters
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Standard LSTM initialization
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
            
    def forward(self, input, hidden):
        h_prev, c_prev = hidden
        
        # Apply the same dropout mask for all timesteps for hidden state
        if self.training and self.dropout > 0:
            h_prev = apply_variational_dropout(h_prev, self.dropout)
            
        # Regular LSTM calculation
        gates = F.linear(input, self.weight_ih, self.bias_ih) + \
                F.linear(h_prev, self.weight_hh, self.bias_hh)
                
        i, f, g, o = gates.chunk(4, 1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class VariationalDropoutLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, batch_first=False):
        super(VariationalDropoutLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Create LSTM cells with variational dropout
        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.lstm_cells.append(VariationalDropoutLSTMCell(layer_input_size, hidden_size, dropout))
    
    def forward(self, input, hidden=None):
        if self.batch_first:
            input = input.transpose(0, 1)  # Convert to seq_len, batch, input_size
            
        seq_len, batch_size, _ = input.size()
        
        if hidden is None:
            h_0 = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            c_0 = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            hidden = (h_0, c_0)
        
        h_n, c_n = [], []
        output = []
        h, c = hidden
        
        # Process the sequence
        for t in range(seq_len):
            inner_input = input[t]
            
            # Process through all LSTM layers
            for layer in range(self.num_layers):
                if layer > 0:
                    # Apply inter-layer dropout if not the first layer
                    if self.training and self.dropout > 0:
                        inner_input = apply_variational_dropout(inner_input, self.dropout)
                
                h_t, c_t = self.lstm_cells[layer](inner_input, (h[layer], c[layer]))
                inner_input = h_t
                
                if t == 0:
                    h_n.append([])
                    c_n.append([])
                
                h_n[layer].append(h_t)
                c_n[layer].append(c_t)
            
            output.append(inner_input)
        
        # Stack the outputs and hidden states
        output = torch.stack(output)
        h_n = torch.stack([torch.stack(h) for h in h_n])
        c_n = torch.stack([torch.stack(c) for c in c_n])
        
        if self.batch_first:
            output = output.transpose(0, 1)  # Convert back to batch, seq_len, hidden
            
        return output, (h_n[-1], c_n[-1])  # Return just the last timestep hidden state

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, dropout=0.1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Use our custom LSTM with variational dropout for hidden states
        self.lstm = VariationalDropoutLSTM(emb_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size, bias=False)  # No bias for weight tying
        # Use the same weights for embedding and output layers
        self.output.weight = self.embedding.weight
        
        # Dropout probability
        self.dropout = dropout
        
    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
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