import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter

# Constants
PAD_TOKEN = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class InputExample:
    """A single training/test example for token classification."""
    def __init__(self, guid, words, intent_label, slot_labels):
        """
        Args:
            guid: Unique id for the example
            words: List of words/tokens in the sentence
            intent_label: The intent label of the sentence
            slot_labels: Slot labels for each word/token
        """
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels


class InputFeatures:
    """Features created from a single sentence for the BERT joint model."""
    def __init__(self, input_ids, attention_mask, token_type_ids, 
                 intent_label, slot_labels, subword_indices):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.subword_indices = subword_indices


def convert_examples_to_features(examples, tokenizer, max_seq_length, 
                                 slot_label_map, intent_label_map, pad_token_label_id=-100):
    """
    Convert examples to features that can be fed to the BERT model
    
    Args:
        examples: InputExample objects
        tokenizer: BERT tokenizer
        max_seq_length: Maximum sequence length
        slot_label_map: Dictionary mapping slot labels to ids
        intent_label_map: Dictionary mapping intent labels to ids
        pad_token_label_id: Label id for padding tokens
        
    Returns:
        List of InputFeatures
    """
    features = []
    
    for (ex_index, example) in enumerate(examples):
        # Convert words to BERT wordpieces
        tokens = []
        slot_label_ids = []
        subword_indices = []  # To keep track of the first subword of each token
        
        word_tokens = [tokenizer.tokenize(word) for word in example.words]
        
        # Flatten and track which indices to use for the original tokens
        curr_token_index = 1  # Start at 1 to account for [CLS]
        for i, word_token_list in enumerate(word_tokens):
            # Record the index of the first subword for each original token
            subword_indices.append(curr_token_index)
            
            # Add all subwords and their labels
            for j, subword in enumerate(word_token_list):
                tokens.append(subword)
                # Only the first subword of a token gets the label
                if j == 0:
                    slot_label_ids.append(slot_label_map[example.slot_labels[i]])
                else:
                    # Use special token for remaining subwords
                    slot_label_ids.append(pad_token_label_id)
                
            curr_token_index += len(word_token_list)
        
        # Add special tokens
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)  # Single sequence, so all 0s
        
        # Pad sequences to max_seq_length
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        else:
            # Truncate if too long
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            token_type_ids = token_type_ids[:max_seq_length]
            # Adjust subword_indices if we truncated
            subword_indices = [idx for idx in subword_indices if idx < max_seq_length - 1]
        
        # Convert intent label to id
        intent_label_id = intent_label_map[example.intent_label]
        
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        
        # Create feature object
        feature = InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            intent_label=intent_label_id,
            slot_labels=slot_label_ids,
            subword_indices=subword_indices
        )
        
        features.append(feature)
    
    return features


class BertJointDataset(Dataset):
    """Dataset for BERT joint intent classification and slot filling"""
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(feature.token_type_ids, dtype=torch.long),
            "intent_label": torch.tensor(feature.intent_label, dtype=torch.long),
            "slot_labels": torch.tensor(feature.slot_labels, dtype=torch.long),
            "subword_indices": feature.subword_indices
        }


def prepare_dataset_from_raw(data_raw, tokenizer, intent_label_map, slot_label_map, max_seq_length=128):
    """
    Prepare dataset for BERT model from raw ATIS data
    
    Args:
        data_raw: Raw ATIS data (list of dictionaries)
        tokenizer: BERT tokenizer
        intent_label_map: Dictionary mapping intent labels to ids
        slot_label_map: Dictionary mapping slot labels to ids
        max_seq_length: Maximum sequence length
        
    Returns:
        BertJointDataset
    """
    examples = []
    for i, item in enumerate(data_raw):
        words = item['utterance'].split()
        intent_label = item['intent']
        slot_labels = item['slots'].split()
        
        # Make sure the number of words matches the number of slot labels
        assert len(words) == len(slot_labels), f"Words and slots length mismatch for example {i}"
        
        example = InputExample(
            guid=f"example-{i}",
            words=words,
            intent_label=intent_label,
            slot_labels=slot_labels
        )
        examples.append(example)
    
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        slot_label_map=slot_label_map,
        intent_label_map=intent_label_map
    )
    
    return BertJointDataset(features)


def bert_collate_fn(batch):
    """
    Collate function for BERT model
    
    Args:
        batch: List of samples from BertJointDataset
        
    Returns:
        Dictionary with batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
    attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch]).to(device)
    intent_label = torch.stack([item["intent_label"] for item in batch]).to(device)
    
    # For slot labels, we need to handle variable lengths
    max_len = max([len(item["subword_indices"]) for item in batch])
    batch_size = len(batch)
    
    # Create padded tensor for slot labels
    slot_labels_padded = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, item in enumerate(batch):
        subword_indices = item["subword_indices"]
        slot_labels = item["slot_labels"]
        
        # Extract the slot labels for the first subword of each token
        for j, idx in enumerate(subword_indices):
            if j < max_len:
                if idx < len(slot_labels):
                    slot_labels_padded[i, j] = slot_labels[idx]
    
    slot_labels_padded = slot_labels_padded.to(device)
    
    # Create a list of subword indices for each item in the batch
    subword_indices = [item["subword_indices"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "intent_label": intent_label,
        "slot_labels": slot_labels_padded,
        "subword_indices": subword_indices
    }


def main():
    # Load ATIS dataset
    train_raw = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join('dataset', 'ATIS', 'test.json'))
    
    # Create a dev set
    portion = 0.10
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)
    
    labels = []
    inputs = []
    mini_train = []
    
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occur only once, we put them in training
            inputs.append(train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(train_raw[id_y])
    
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=portion, 
        random_state=42, 
        shuffle=True,
        stratify=labels
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    
    # Get all slot and intent labels
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    
    # Create label maps
    slot_label_map = {label: i for i, label in enumerate(slots)}
    slot_label_map['pad'] = PAD_TOKEN  # Add padding token
    
    intent_label_map = {label: i for i, label in enumerate(intents)}
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = prepare_dataset_from_raw(
        train_raw, tokenizer, intent_label_map, slot_label_map
    )
    dev_dataset = prepare_dataset_from_raw(
        dev_raw, tokenizer, intent_label_map, slot_label_map
    )
    test_dataset = prepare_dataset_from_raw(
        test_raw, tokenizer, intent_label_map, slot_label_map
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=bert_collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=bert_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=bert_collate_fn
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Example usage of a batch
    for batch in train_loader:
        print("Batch shape:")
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Attention Mask: {batch['attention_mask'].shape}")
        print(f"Token Type IDs: {batch['token_type_ids'].shape}")
        print(f"Intent Label: {batch['intent_label'].shape}")
        print(f"Slot Labels: {batch['slot_labels'].shape}")
        break


if __name__ == "__main__":
    main()