
from transformers import BertConfig
from model import  DNAModel

import argparse
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig
import torch, os
from datasets import load_dataset
from utils import save_result, tokenization_dataset, train_tokenizer
from transformers import DistilBertConfig
def train_model(model: torch.nn.Module, opt, dataloaders, tokenizer):
    for (train_dataloader, test_dataloader) in dataloaders:
        configuration = DistilBertConfig(vocab_size=256, model_type="regression", num_labels=1, n_layers = 6)
        model = DNAModel(configuration)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        train_model_single(model, opt, train_dataloader, test_dataloader, tokenizer)

def train_model_single(model: torch.nn.Module, opt, train_dataloader, eval_dataloader, tokenizer):
    from transformers import AdamW, Adafactor, get_scheduler
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-5)
    num_epochs = opt.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))
    writer = SummaryWriter()
    
    model.train()
    for epoch in range(num_epochs):
        all_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            all_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        all_loss /= len(train_dataloader)
        eval_loss = eval_model(model, eval_dataloader, opt, epoch)
        writer.add_scalar("Loss/train",all_loss , epoch)
        writer.add_scalar("Loss/test", eval_loss , epoch)
        # save_result(best_model, eval_dataloader, opt, True, epoch)
        torch.save(best_model.state_dict(), f"{opt.log_dir}/{epoch}.dict")
    save_result(best_model, eval_dataloader, opt, False, epoch, tokenizer)
    torch.save(best_model.state_dict(), f"{opt.log_dir}/transformer")
    writer.flush()
    writer.close()

best_loss = 1e9
best_model= None
def eval_model(model: torch.nn.Module, eval_dataloader, opt, epoch):
    import copy
    model.eval()
    global best_loss, best_model
    with torch.no_grad():
        all_loss = 0.0
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
                #print(f'batch is {batch["labels"]}')
            outputs = model(**batch)
            loss = outputs.loss.item()
            all_loss += loss
        all_loss /= len(eval_dataloader)
        if all_loss < best_loss:
            best_loss = all_loss
            best_model = copy.deepcopy(model)
                #save_result(best_model, eval_dataloader, opt, False, epoch)
    model.train()
    return all_loss  

def initialize(parser):
    parser.add_argument('--mode', required=True, choices=['Train', 'Eval', 'Transfer'],
                            help='Train | Eval')
    parser.add_argument('--save_dir', type=str, default='./prev.csv')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--log_dir', required=True, type=str)
    parser.add_argument('--pretrained_weights', type=str)
    opt, _ = parser.parse_known_args()
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    return opt

def transfer_model(model, opt, dataloaders):
    load_state_dict(model, torch.load(f"{opt.logs}/transformer"))
    train_model(model, opt, dataloaders)

def load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in ('left', 'right'):
            own_state[name].copy(param.data)

def pre_process(opt):
    seed = 1997
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # Accessing the model configuration
    dataroot = opt.dataroot
    data_files = {"train": f"{dataroot}/train.csv"}

    #print(dataset)
    
    trains_ds = load_dataset('csv', data_files=data_files, split=[
        f'train[:{k}%]+train[{k+20}%:]' for k in range(0, 100, 20)
    ])
    vals_ds = load_dataset('csv', data_files=data_files, split=[
        f'train[{k}%:{k+20}%]' for k in range(0, 100, 20)
    ])

    tokenizer = train_tokenizer(trains_ds, vals_ds)
    configuration = DistilBertConfig(vocab_size=256, model_type="regression", num_labels=1, n_layers = 6)
    model = DNAModel(configuration)

    return model, [tokenization_dataset(train_ds, test_ds, tokenizer) for train_ds, test_ds in zip(trains_ds, vals_ds)], tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opt = initialize(parser)
    model, dataloaders, tokenizer = pre_process(opt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    if opt.mode == 'Train':
        train_model(model, opt, dataloaders, tokenizer)    
    elif opt.mode == 'Transfer':
        transfer_model(model, opt, dataloaders)
    else:
        save_result(model, dataloaders, opt, True, "prev")
        print("done")