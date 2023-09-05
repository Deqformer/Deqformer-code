import torch          
from tokenizers import NormalizedString, PreTokenizedString
from typing import List
import csv, os

from tokenizers.trainers import WordLevelTrainer
from datasets.splits import Split
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from torch.utils.data import DataLoader

class DNAPreTokenizer:
    def dna_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        k_mer_tokenizer = lambda k : lambda word : [str(word[i:i+k]) for i in range(len(word) - k + 1)]
        for token in k_mer_tokenizer(3)(str(normalized_string)):
            splits.append(NormalizedString(token))
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.dna_split)


def clean_dataset(dataset, tokenizer):
    def tokenize_function(input):
        encode_ids = tokenizer.encode(str(input['text'])).ids
        return {'input_ids': encode_ids, 'rev_input_ids': encode_ids}
    tokenized_datasets = dataset.map(tokenize_function)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.remove_columns(["rev_text"])
    tokenized_datasets = tokenized_datasets.remove_columns(["id"])
    tokenized_datasets = tokenized_datasets.rename_column("depth", "labels")
    tokenized_datasets.set_format("torch")
    return tokenized_datasets

def train_tokenizer(train_data, test_data):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = PreTokenizer.custom(DNAPreTokenizer())
    trainer = WordLevelTrainer(vocab_size=256, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    data= [e['text'] for e in train_data]
    data.extend([e['text'] for e in test_data])
    data.extend([e['rev_text'] for e in train_data])
    data.extend([e['rev_text'] for e in test_data])
    tokenizer.train_from_iterator(data, trainer=trainer)
    return tokenizer

def tokenization_dataset(train_data, test_data):
    tokenizers = train_tokenizer(train_data, test_data)

    full_train_dataset = clean_dataset(train_data, tokenizers)
    full_eval_dataset = clean_dataset(test_data, tokenizers)

    train_dataloader = DataLoader(full_train_dataset, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(full_eval_dataset, batch_size=16)
    return (train_dataloader, eval_dataloader)
