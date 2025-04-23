from transformers import BertModel
import torch.nn as nn
import torch

class StandardBERT(nn.Module):
    def __init__(self, num_intents, num_slots):
        super(StandardBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        hidden_size = self.bert.config.hidden_size
        
        # Simple intent classification head
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        
        # Simple slot classification head
        self.slot_classifier = nn.Linear(hidden_size, num_slots)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.zeros_(self.intent_classifier.bias)
        nn.init.xavier_uniform_(self.slot_classifier.weight)
        nn.init.zeros_(self.slot_classifier.bias)
        
        # Freeze BERT embeddings to stabilize training
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
    
    def forward(self, token_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        
        # Get sequence output for slots and pooled output for intent
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Apply dropout to both
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_logits, slot_logits