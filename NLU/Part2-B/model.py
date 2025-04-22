from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from config import dropout

class PositionalEncoding(nn.Module):
    """Position encoding for enhancing token position awareness"""
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input tensor"""
        x = x + self.pe[:x.size(1), :]
        return x

class StandardBERT(nn.Module):
    def __init__(self, num_intents, num_slots):
        super(StandardBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        hidden_size = self.bert.config.hidden_size
        
        # Intent classification with enhanced architecture
        self.intent_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        
        # Add positional encoding for better position awareness in slots
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Bidirectional Hierarchical LSTM for slot classification
        self.slot_lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        self.slot_layernorm = nn.LayerNorm(512)  # 512 = 256*2 (bidirectional)
        
        self.slot_lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Residual connection and layer norm for slot prediction
        self.slot_residual_proj = nn.Linear(hidden_size, 512)
        self.slot_layernorm_final = nn.LayerNorm(512)
        
        # Enhanced slot classification head
        self.slot_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_slots)
        )
        
        # Store number of slots
        self.slots = num_slots
        
        # Better initialization
        self._init_weights()
        
        # Selective unfreezing - only train task-specific layers and last 6 BERT layers
        self._freeze_bert_layers(8)  # Freeze first 8 layers
    
    def _init_weights(self):
        """Initialize weights with better strategies for different components"""
        # Intent classifier
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.zeros_(self.intent_classifier.bias)
        
        # Slot classifier layers
        nn.init.xavier_uniform_(self.slot_classifier[0].weight)
        nn.init.zeros_(self.slot_classifier[0].bias)
        nn.init.xavier_uniform_(self.slot_classifier[4].weight)
        nn.init.zeros_(self.slot_classifier[4].bias)
        
        # Residual projection
        nn.init.xavier_uniform_(self.slot_residual_proj.weight)
        nn.init.zeros_(self.slot_residual_proj.bias)
    
    def _freeze_bert_layers(self, num_freeze):
        """Freeze the first `num_freeze` layers of BERT"""
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:num_freeze]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self, token_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        
        # Intent classification from [CLS] token
        pooled_output = outputs.pooler_output
        pooled_output = self.intent_ffn(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        
        # Slot classification from sequence tokens
        sequence_output = outputs.last_hidden_state
        sequence_output = self.positional_encoding(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        # Apply hierarchical LSTM to BERT outputs
        lstm_output1, _ = self.slot_lstm1(sequence_output)
        lstm_output1 = self.slot_layernorm(lstm_output1)
        lstm_output2, _ = self.slot_lstm2(lstm_output1)
        
        # Residual connection and final layer norm
        residual = self.slot_residual_proj(sequence_output)
        lstm_output2 += residual
        lstm_output2 = self.slot_layernorm_final(lstm_output2)
        
        # Get slot logits from LSTM output
        slot_logits = self.slot_classifier(lstm_output2)
        
        return intent_logits, slot_logits