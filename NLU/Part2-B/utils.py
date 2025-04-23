import json
from typing import Counter
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os


class IntentsAndSlots(Dataset):
    def __init__(self, dataset, lang, unk='unk', tokenizer=None, max_len=50, myType=None):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.lang = lang
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.intent_ids = self.mapping_labels(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = self.utterances[idx]
        slot_labels = self.slots[idx].split()
        intent = self.intent_ids[idx]
        
        # Tokenize with special tokens
        encoding = self.tokenizer(
            utt,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        
        # Extract tensor components and remove batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        special_tokens_mask = encoding['special_tokens_mask'].squeeze(0)
        
        # Prepare slot labels
        words = utt.split()
        word_ids = encoding.word_ids(batch_index=0)
        
        # Initialize slot IDs for the entire sequence with padding
        slots = []
        
        for word_id in word_ids:
            # Special tokens get pad token
            if word_id is None:
                slots.append(self.lang.slot2id['pad'])
            # Regular words get their corresponding slot label
            elif word_id < len(slot_labels):
                slots.append(self.lang.slot2id.get(slot_labels[word_id], self.lang.slot2id['O']))
            # Words beyond our slot labels (shouldn't happen) get O
            else:
                slots.append(self.lang.slot2id['O'])
        
        # Create sample dictionary
        sample = {
            'utterance': utt,  # Keep original utterance for debugging
            'utterances': input_ids,  # For model input
            'attention_mask': attention_mask,
            'slots': torch.tensor(slots),
            'intent': intent
        }
        
        return sample
    
    # Auxiliary methods
    def mapping_labels(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]


# Loading the corpus
def load_data(path):
    from json import loads
    dataset = []
    with open(path) as f:
        dataset = loads(f.read())
    return dataset


def load_from_local_atis(data_dir='dataset/ATIS'):
    """
    Load ATIS dataset from local directory, creating dev set from train set.
    Returns a dictionary with 'train', 'test', and 'dev' splits.
    """
    try:
        # Try to load train and test data
        train_path = os.path.join(data_dir, 'train.json')
        test_path = os.path.join(data_dir, 'test.json')
        
        train_raw = []
        test_raw = []
        
        with open(train_path) as f:
            train_raw = json.loads(f.read())
        
        with open(test_path) as f:
            test_raw = json.loads(f.read())
        
        # Create dev set from train set following Part2-A approach
        portion = 0.10
        
        intents = [x['intent'] for x in train_raw]  # We stratify on intents
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
        
        atis_data = {
            'train': train_raw,
            'test': test_raw,
            'dev': dev_raw
        }
        
        # Add this immediately after loading the dataset
        print("Dataset loaded:", bool(atis_data))
        if 'train' in atis_data:
            print("Sample utterance:", atis_data['train'][0]['utterance'] if len(atis_data['train']) > 0 else "No samples")
        
        return atis_data
        
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Make sure the ATIS dataset files are in the correct location.")
        return {}


def generate_validation_set(training_set_raw, percentage=0.1):
    from collections import Counter

    intents = [x['intent'] for x in training_set_raw]
    count_intents = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for idx, intent in enumerate(intents):
        # if intent occurs only once, put it in train
        if(count_intents[intent] > 1):
            inputs.append(training_set_raw[idx])
            labels.append(intent)
        else: #else put it in val
            mini_train.append(training_set_raw[idx])

    x_train, x_val, intent_train, intent_val = train_test_split(inputs, labels, test_size=percentage, random_state=42, shuffle=True, stratify=labels)

    x_train.extend(mini_train)
    train_raw = x_train
    val_raw = x_val


    ''' train_raw[0]= {'intent': 'airfare',
                        'slots': 'O O O O O O O O B-fromloc.city_name O B-toloc.city_name',
                        'utterance': 'what is the cost for these flights from baltimore to '
                                     'philadelphia'
                       }
        y_train[0] = intent of train_raw[0]
            
        val_raw[0] is same as train but for the validation set (generated one)
        y_val[0] = intent of val_raw[0]

        test_raw[0] same as the other two but for test set
        y_test[0] = intent of test_raw[0]
    
    '''

    # Intent distributions
    # print('Train:')
    # pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    # print('Dev:'), 
    # pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    # print('Test:') 
    # pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    # print('='*89)
    # # Dataset size
    # print('TRAIN size:', len(train_raw))
    # print('DEV size:', len(dev_raw))
    # print('TEST size:', len(test_raw))

    return train_raw, intent_train, val_raw, intent_val