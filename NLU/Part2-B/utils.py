import torch
from torch.utils.data import Dataset
from subprocess import run


class IntentsAndSlots(Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
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

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_labels(self.intents, lang.intent2id)


    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        #utt = torch.Tensor(self.utt_ids[idx])
        utt = self.utterances[idx]
        #utt_ids = self.utt_ids[idx]
        slots_ids = self.slot_ids[idx]
        intent = self.intent_ids[idx]
        #slots_labels = self.slots[idx]


        # Tokenize the utterance into words
        words = utt.split()
        word_slots = []
        for word, slot in zip(words, slots_ids):
            subwords = self.tokenizer.tokenize(word)
            word_slots.extend([slot] * len(subwords))

        '''text_encoding = self.tokenizer.encode_plus(utt,
                                                   max_length=self.max_len,
                                                   add_special_tokens=True,
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_attention_mask=True,
                                                   return_tensors='pt')
        

        token = self.tokenizer.tokenize(utt)
        token_ids = text_encoding['input_ids'].flatten()
        attention_mask = text_encoding['attention_mask'].flatten()'''

        inputs = self.tokenizer(utt,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len,
                                return_tensors='pt')
        

        inputs_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        #tokens = self.tokenizer.tokenize(utt)

        aligned_labels = [self.lang.slot2id['O']] + word_slots + [self.lang.slot2id['O']]

        while len(aligned_labels) < len(inputs_ids):
            aligned_labels.append(self.lang.slot2id['pad'])

        aligned_labels = aligned_labels[:len(inputs_ids)]  # Ensure alignment

        
        sample = {'utterance': inputs_ids,
                  'attention_mask': attention_mask,
                  'slots': torch.tensor(aligned_labels),
                  'intent': intent
                  }

        
        '''sample = {'utterance': token_ids,
                  'attention_mask': torch.Tensor(attention_mask),
                  #'tokenizer': self.tokenizer,
                  'original_utterance': utt,
                  #'intent': torch.tensor(intent),
                  #'slots': torch.tensor(slots)
                  'intent': intent,
                  'slots': torch.Tensor(encoded)
                 }'''
        return sample
    
    # Auxiliary methods
    
    def mapping_labels(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    

# Loading the corpus
def load_data(path):
    from json import loads
    dataset = []
    with open(path) as f:
        dataset = loads(f.read())
    return dataset


def load_from_local_atis(data_dir='dataset/ATIS'):
    """
    Load data from the local ATIS dataset directory
    Args:
        data_dir: Path to the ATIS dataset directory
    Returns:
        Dictionary containing training, validation and test datasets
    """
    import os
    from json import loads
    
    data = {}
    
    # Map of expected files in the ATIS directory
    file_mapping = {
        'train': 'atis.train.json',
        'valid': 'atis.dev.json',
        'test': 'atis.test.json'
    }
    
    # Load each dataset file
    for split, filename in file_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[split] = loads(f.read())
            print(f"Loaded {split} set with {len(data[split])} examples")
        else:
            print(f"Warning: {filepath} not found")
    
    return data


def generate_validation_set(training_set_raw, percentage=0.1):
    from collections import Counter
    from sklearn.model_selection import train_test_split

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