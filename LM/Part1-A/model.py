import torch.nn as nn

# LSTM with dropout layers
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                    emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index) # Embedding layer
        self.emb_dropout = nn.Dropout(emb_dropout)  # Dropout after embedding

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.output_dropout = nn.Dropout(out_dropout)  # Dropout before final linear layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)  # Embedding layer
        emb = self.emb_dropout(emb)  # Apply dropout after embedding

        lstm_out, _ = self.lstm(emb)  # LSTM layer

        lstm_out = self.output_dropout(lstm_out)  # Apply dropout before linear layer
        output = self.output(lstm_out).permute(0, 2, 1)  # Final output

        return output