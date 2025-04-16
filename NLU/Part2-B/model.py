from transformers import BertModel
import torch.nn as nn
from config import dropout

class StandardBERT(nn.Module):
    def __init__(self, num_intents, num_slots):
        super(StandardBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        
        # Intent classification head
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        
        # Slot classification head
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)
        
        # Store number of slots for validation function
        self.slots = num_slots
    
    def forward(self, token_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        
        # Get the pooled output for intent classification (CLS token)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        
        # Get the sequence output for slot classification
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_logits, slot_logits