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

# def train_tokenizer(train_data, test_data):
#     tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
#     tokenizer.pre_tokenizer = PreTokenizer.custom(DNAPreTokenizer())
#     trainer = WordLevelTrainer(vocab_size=256, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
#     data= [e['text'] for e in train_data]
#     data.extend([e['text'] for e in test_data])
#     data.extend([e['rev_text'] for e in train_data])
#     data.extend([e['rev_text'] for e in test_data])
#     tokenizer.train_from_iterator(data, trainer=trainer)
#     return tokenizer

def tokenization_dataset(train_data, test_data, tokenizers):

    full_train_dataset = clean_dataset(train_data, tokenizers)
    full_eval_dataset = clean_dataset(test_data, tokenizers)

    train_dataloader = DataLoader(full_train_dataset, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(full_eval_dataset, batch_size=16)
    return (train_dataloader, eval_dataloader)

def train_tokenizer(train_datas, test_datas):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = PreTokenizer.custom(DNAPreTokenizer())
    trainer = WordLevelTrainer(vocab_size=256, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    #print(f'train_data is {train_data.next()}')
    for (train_data, test_data) in zip(train_datas, test_datas):
        data= [e['text'] for e in train_data]
        # data.extend([e['text'] for e in test_data])
        data.extend([e['rev_text'] for e in train_data])
        #data.extend([e['rev_text'] for e in test_data])
        tokenizer.train_from_iterator(data, trainer=trainer)
    return tokenizer

def save_result(model: torch.nn.Module, train_dataloader, opt, load_model, epoch, tokenzier) -> float:
    des_dir = f'{opt.log_dir}/result.csv'
    train_dataloader = train_dataloader
    if load_model:
        #des_dir = f'{opt.log_dir}/{epoch}.csv'
        model.load_state_dict(torch.load(f"{opt.log_dir}/transformer"))
    model.eval()  # turn on evaluation mode
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.no_grad():
        with open(des_dir, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file,delimiter=',')
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                #print(f'batch is {batch}')
                outputs = model(**batch)
                for e in range(outputs.logits.shape[0]):
                    pred = outputs.logits[e].item()
                    real = batch["labels"][e].item()
                    id = batch["input_ids"][e].tolist()
                    id = tokenzier.decode(id).split()
                    # mean = 6.599401777846243
                    # std = 1.2772261792750987
                    std = 0.41257507475041016 
                    mean =  2.7706192889195442
                    pred = pred * std + mean
                    real = real * std + mean
                    writer.writerow([pred , real])
                    #print(outputs.logits[e].item(), batch["labels"][e].item())
                #break
    model.train()
    