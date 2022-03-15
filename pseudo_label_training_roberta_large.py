import warnings
warnings.filterwarnings("ignore")


import gc
import os
import re
import ast
import time
import copy
import torch
import json
import wandb
import joblib
import random
import itertools
import numpy as np
import pandas as pd
import transformers
from torch import nn 
from tqdm.notebook import tqdm
import torch.nn.functional as F 
from sklearn import metrics, model_selection
from torch.utils.data import Sampler, Dataset, DataLoader

gc.enable()

config = dict(
    # basic
    seed = 3407,
    num_jobs=1,
    num_labels=2,
    num_folds=5,
    
    # model info
    tokenizer_path = '../input/robertalarge',
    model_checkpoint = '../input/robertalarge', 
    resume_training_checkpoint = None,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # trining paramters
    learning_rate = 1e-5,
    weight_decay = 1e-2,
    max_length = 420,
    train_batch_size = 4,
    valid_batch_size = 8,
    epochs_to_train = 5,
    total_epochs = 5,
    grad_acc_steps = 4,
    
    # for this notebook
    report_to = 'wandb',
    output_dir = '',
    fold_to_train = [1],
    title = 'roberta-large-multidrop-pseudolabel',
    debug = False,
    platform = 'kaggle', # kaggle, colab
    inference_only = False,
)

title = config['title']

if config['platform'] == 'colab':
    config['output_dir'] = f'../output/{title}/'
    base_path = 'drive/MyDrive/NBME'
    os.chdir(base_path + '/src')

def setup_wandb(name):
    if config['platform'] == 'kaggle':
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_token")
    else:
        secret_value_0 = '...' 

    wandb.login(key=secret_value_0)
    wandb.init(
        project='NBME - Score Clinical Patient Notes',
        entity="mananjhaveri",
        name=name,
        save_code=True,
    )
    wandb.config = config

def create_folds(data):
    
    data['kfold'] = -1
    data['for_stratify'] = data['case_num'].astype(str) + '_' + data['feature_num'].astype(str)

    kf = model_selection.StratifiedKFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['for_stratify'])):
        data.loc[v_, 'kfold'] = f
    
    data.drop(['for_stratify'], axis=1, inplace=True)

    return data

def loc_list_to_ints(loc_list):
    to_return = []
    for loc_str in loc_list:
        loc_strs = loc_str.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
    return to_return


def tokenize_and_add_labels(tokenizer, example):
    tokenized_inputs = tokenizer(
        example["feature_text"],
        example["pn_history"],
        truncation="only_second",
        max_length=config['max_length'],
        padding="max_length",
        return_offsets_mapping=True
    )
    labels = [0.0] * len(tokenized_inputs["input_ids"])
    tokenized_inputs["location_int"] = loc_list_to_ints(example["location"])
    tokenized_inputs["sequence_ids"] = tokenized_inputs.sequence_ids()
    
    for idx, (seq_id, offsets) in enumerate(zip(tokenized_inputs["sequence_ids"], tokenized_inputs["offset_mapping"])):
        if seq_id is None or seq_id == 0:
            labels[idx] = -100
            continue
        exit = False
        token_start, token_end = offsets
        for feature_start, feature_end in tokenized_inputs["location_int"]:
            if exit:
                break
            if token_start >= feature_start and token_end <= feature_end:
                labels[idx] = 1.0
                exit = True
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs

class NBMEData(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data.loc[idx]
        tokenized = tokenize_and_add_labels(self.tokenizer, example)
        
        input_ids = np.array(tokenized["input_ids"])
        attention_mask = np.array(tokenized["attention_mask"])
        labels = np.array(tokenized["labels"])
        offset_mapping = np.array(tokenized["offset_mapping"])
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16")
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'targets': labels, 
            'offset_mapping': offset_mapping, 
            'sequence_ids': sequence_ids,
        }

class NBMEModel(nn.Module):
    def __init__(self, num_labels, path=None):
        super().__init__()
        
        layer_norm_eps: float = 1e-6
        
        self.path = path
        self.num_labels = num_labels
        self.config = transformers.AutoConfig.from_pretrained(config['model_checkpoint'])

        self.config.update(
            {
                "layer_norm_eps": layer_norm_eps,
            }
        )
        self.transformer = transformers.AutoModel.from_pretrained(config['model_checkpoint'], config=self.config)
        self.dropout = nn.Dropout(0.1)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(self.config.hidden_size, 1)
        
        if self.path is not None:
            self.load_state_dict(torch.load(self.path)['model'])
    
    def forward(self, data):
        
        ids = data['input_ids']
        mask = data['attention_mask']
        try:
            target = data['targets']
        except:
            target = None

        transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out[0]
        sequence_output = self.dropout(sequence_output)
    
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        ret = {
            'logits': torch.sigmoid(logits), 
        }
        
        loss = 0

        if target is not None:
            loss1 = self.get_loss(logits1, target)
            loss2 = self.get_loss(logits2, target)
            loss3 = self.get_loss(logits3, target)
            loss4 = self.get_loss(logits4, target)
            loss5 = self.get_loss(logits5, target)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            ret['loss'] = loss
            ret['target'] = target

        return ret

        
    def get_optimizer(self, learning_rate, weigth_decay):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weigth_decay,
        )
        if self.path is not None:
            optimizer.load_state_dict(torch.load(self.path)['optimizer'])
        
        return optimizer
            
    def get_scheduler(self, optimizer, num_warmup_steps, num_training_steps):
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        if self.path is not None:
            scheduler.load_state_dict(torch.load(self.path)['scheduler'])
            
        return scheduler
    
    def get_loss(self, output, target):
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fn(output.view(-1, 1), target.view(-1, 1))
        loss = torch.masked_select(loss, target.view(-1, 1) != -100).mean()
        return loss

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def train_fn(model, train_loader, optimizer, scheduler, device, current_epoch):  
    losses = AverageMeter()
    optimizer.zero_grad()

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, data in enumerate(tepoch):
            for k, v in data.items():
                if k != 'offset_mapping':
                    data[k] = v.to(config['device'])

            model.train()
            loss = model(data)['loss'] / config['grad_acc_steps']
                
            loss.backward()
            losses.update(loss.item(), len(train_loader))
            tepoch.set_postfix(train_loss=losses.avg)
            
            if batch_idx % config['grad_acc_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
                if config['report_to'] == 'wandb':
                    wandb.log({"epoch": current_epoch, "train_loss": losses.avg, 'lr': scheduler.get_lr()[0]})
                    
            
def eval_fn(model, valid_loader, device, current_epoch):
    losses = AverageMeter()

    final_targets = []
    final_predictions = []
    offset_mapping = []
    sequence_ids = []

    model.eval()
    
    with torch.no_grad():
        
        with tqdm(valid_loader, unit="batch") as tepoch:

            for batch_idx, data in enumerate(tepoch):
                for k, v in data.items():
                    if k not in  ['offset_mapping', 'sequence_id']:
                        data[k] = v.to(config['device'])
                
                x = model(data)
                loss = x['loss']
                losses.update(loss.item(), len(valid_loader))

                o = x['logits'].detach().cpu().numpy()
                final_predictions.extend(o)
                
                y = data['targets'].detach().cpu().numpy()
                final_targets.extend(y)
                
                offset_mapping.extend(data['offset_mapping'].tolist())
                sequence_ids.extend(data['sequence_ids'].tolist())
    
    predicted_locations = decode_predictions(final_predictions, offset_mapping, sequence_ids, test=False)
    scores = get_score(predicted_locations, offset_mapping, sequence_ids, final_targets)

    if config['report_to'] == 'wandb':
        wandb.log({"epoch": current_epoch, "val_loss": losses.avg, 'val_score': scores['f1']})

    return round(losses.avg, 4), round(scores['f1'], 4)

def decode_predictions(preds, offset_mapping, sequence_ids, test=False):
    
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        start_idx = None
        current_preds = []
        
        for p, o, s_id in zip(pred, offsets, seq_ids):
            
            # do nothing if sequence id is not 1
            if s_id is None or s_id == 0:
                continue
                
            # if class = 1, track start and end location from offset map
            if p > 0.5:
                if start_idx is None:
                    start_idx = o[0]
                end_idx = o[1]
            
            # if class 0, record previously tracked predictions if not done already
            elif start_idx is not None:
                if test:
                    current_preds.append(f"{start_idx} {end_idx}")
                else:
                    current_preds.append((start_idx, end_idx))
                start_idx = None # reset
                
        if test:
            all_predictions.append("; ".join(current_preds)) # submission format requirement
        else:
            all_predictions.append(current_preds)
            
    return all_predictions


def get_score(predictions, offset_mapping, sequence_ids, labels):
    all_labels = []
    all_preds = []
    
    for preds, offsets, seq_ids, labels in zip(predictions, offset_mapping, sequence_ids, labels):
        num_chars = max(list(itertools.chain(*offsets)))
        char_labels = np.zeros((num_chars))
        
        # formatting ground truth for evaluation
        for o, s_id, label in zip(offsets, seq_ids, labels):
            # do nothing if sequence id is not 1
            if s_id is None or s_id == 0:
                continue
            if int(label) == 1:
                char_labels[o[0]: o[1]] = 1
            
        # formatting predictions for evaluation
        char_preds = np.zeros((num_chars))
        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
            
        all_labels.extend(char_labels)
        all_preds.extend(char_preds)
        
    results = metrics.precision_recall_fscore_support(all_labels, all_preds, average = "binary")
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }

def save_checkpoint(model, optimizer, scheduler, epoch, score, best_score, name):
    print('saving model of this epoch as:', name)
    
    name = config['output_dir'] + name
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'score': score,
            'best_score': best_score,
        },
        name
    )

def run(df, fold, tokenizer, device, resume_training_checkpoint=None):

    print('Fold:', fold)

    print('\npreparing training data...')
    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    train_dataset = NBMEData(train_df, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
    )
    
    print('\npreparing validation data...')
    valid_df = df[df['kfold'] == fold].reset_index(drop=True)
    valid_dataset = NBMEData(valid_df, tokenizer)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['valid_batch_size'],
        shuffle=False,
    )

    model = NBMEModel(config['num_labels'], resume_training_checkpoint)
    model.to(device)

    num_training_steps = (len(train_dataset) // (config['train_batch_size'] * config['grad_acc_steps'])) * config['total_epochs']
    num_warmup_steps = int(num_training_steps * 0.01)
    optimizer = model.get_optimizer(config['learning_rate'], config['weight_decay'])
    scheduler = model.get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    config['num_training_steps'] =  num_training_steps
    config['num_warmup_steps'] =  num_warmup_steps 

    if config['report_to'] == 'wandb':
        setup_wandb(config['title'] + '-' + str(fold))
        wandb.watch(model, log_freq=10)

    epoch_start = 0
    best_score = -1
    if resume_training_checkpoint is not None:
        epoch_start = torch.load(resume_training_checkpoint)['epoch'] + 1
        best_score = torch.load(resume_training_checkpoint)['best_score']
    start = time.time()

    for epoch in range(epoch_start, epoch_start + config['epochs_to_train']):   

        print(f'\n\n\nTraining Epoch: {epoch}')
        train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        
        print('Evaluation...')
        val_loss, val_score = eval_fn(
            model=model, 
            valid_loader=valid_loader, 
            device=device,
            current_epoch=epoch,
        )
        
        if val_score > best_score:
            best_score = val_score
            save_checkpoint(model, optimizer, scheduler, epoch, val_score, best_score, f'best_model_{fold}.bin')

        save_checkpoint(model, optimizer, scheduler, epoch, val_score, best_score, f'last_model_{fold}.bin')

        print('Valid Score:', val_score, 'Valid Loss:', val_loss, 'Best Score:', best_score)
        
    print(f'Best Score: {best_score}, Time Taken: {round(time.time() - start, 4)}s')
    print()
    
    if config['report_to'] == 'wandb':    
        wandb.finish()

tokenizer = transformers.AutoTokenizer.from_pretrained(config['tokenizer_path'])

train = pd.read_csv('../input/nbme-cleaned-with-extra-data-and-folds/train.csv')
notes = pd.read_csv('../input/nbme-score-clinical-patient-notes/patient_notes.csv')
feats = pd.read_csv('../input/nbme-score-clinical-patient-notes/features.csv')

df_pseudo_label = pd.read_csv(
    '../input/deberta-v3-large-0-883-lb-pseudo-label/submission.csv'
)
df_pseudo_label, _ = model_selection.train_test_split(
    df_pseudo_label,
    test_size=0.6,
    stratify=df_pseudo_label['feature_num'].astype(str) + '_' + df_pseudo_label['case_num'].astype(str),
    random_state=config['seed'],
)
df_pseudo_label['kfold'] = -1
df_pseudo_label.drop(['pn_history', 'feature_text'], axis=1)
df_pseudo_label = df_pseudo_label.merge(notes, how = "left")
df_pseudo_label = df_pseudo_label.merge(feats, how = "left")

train = pd.concat([train, df_pseudo_label]).reset_index(drop=True)
train['location'] = train['location'].apply(ast.literal_eval)

if config['debug']:
    train = train.sample(config['debug']).reset_index(drop=True)

if not config['inference_only']:
    for fold in config['fold_to_train']:
        run(
            df=train, 
            fold=fold,
            tokenizer=tokenizer,
            device=config['device'],
            resume_training_checkpoint=config['resume_training_checkpoint'],
        )
        torch.cuda.empty_cache()
        gc.collect()