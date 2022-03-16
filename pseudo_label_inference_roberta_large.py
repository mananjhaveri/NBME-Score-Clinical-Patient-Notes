import warnings
warnings.simplefilter('ignore')

import os
import gc

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange

import torch
from torch import nn
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig


# config

ROOT = '../input/nbme-score-clinical-patient-notes'
config = dict(
    # basic
    seed = 3407,
    num_jobs=1,
    num_labels=2,
    num_folds=5,
    
    # model info
    tokenizer_path = '../input/robertalarge', # 'roberta-base', 
    model_checkpoint = '../input/robertalarge', # 'roberta-base', 
    resume_training_checkpoint = None,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # trining paramters
    max_length = 512,
    valid_batch_size = 32,
    
    # for this notebook
    platform = 'kaggle', # kaggle, colab
)

def create_test_df():
    feats = pd.read_csv(f"{ROOT}/features.csv")
    feats.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    
    notes = pd.read_csv(f"{ROOT}/patient_notes.csv")
    test = pd.read_csv(f"{ROOT}/test.csv")

    merged = test.merge(notes, how = "left")
    merged = merged.merge(feats, how = "left")

    def process_feature_text(text):
        return text.replace("-OR-", ";-").replace("-", " ")
    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    
    return merged

class NBMETestData(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data.loc[idx]
        tokenized = self.tokenizer(
            example["feature_text"],
            example["pn_history"],
            truncation = "only_second",
            max_length = config['max_length'],
            padding = "max_length",
            return_offsets_mapping = True
        )
        tokenized["sequence_ids"] = tokenized.sequence_ids()

        input_ids = np.array(tokenized["input_ids"])
        attention_mask = np.array(tokenized["attention_mask"])
        offset_mapping = np.array(tokenized["offset_mapping"])
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16")

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'offset_mapping': offset_mapping, 
            'sequence_ids': sequence_ids,
        }

class NBMEModel(nn.Module):
    def __init__(self, num_labels=2, path=None):
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


def get_location_predictions(preds, offset_mapping, sequence_ids, threshold=0.5, test=False):
    all_predictions = []
    
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        start_idx = None
        current_preds = []
        
        for p, o, s_id in zip(pred, offsets, seq_ids):
            
            if s_id is None or s_id == 0:
                continue
                
            if p > threshold:
                if start_idx is None:
                    start_idx = o[0]
                end_idx = o[1]
                
            elif start_idx is not None:
                if test:
                    current_preds.append(f"{start_idx} {end_idx}")   
                else:
                    current_preds.append((start_idx, end_idx))   
                start_idx = None
                
        if test:
            all_predictions.append("; ".join(current_preds))
        else:
            all_predictions.append(current_preds)
            
    return all_predictions

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

test = create_test_df()
test_ds = NBMETestData(test, tokenizer)
test_dl = torch.utils.data.DataLoader(
    test_ds, 
    batch_size=config['valid_batch_size'], 
    pin_memory=True, 
    shuffle=False, 
    drop_last=False
)

paths = [
    '../input/nbme-training-roberta-large-pseudo-label-f0/best_model_0.bin',
    '../input/nbme-training-roberta-large-pseudo-label-f1/best_model_1.bin',
    '../input/nbme-training-roberta-large-pseudo-label-f2/best_model_2.bin',
    '../input/nbme-training-roberta-large-pseudo-label-f3/best_model_3.bin',
    '../input/nbme-training-roberta-large-pseudo-label-f4/best_model_4.bin',
]


all_preds = None
offsets = []
seq_ids = []

for model_no, path in enumerate(paths):
    model = NBMEModel().to(config['device'])
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl):
            
            for k, v in batch.items():
                if k not in  ['offset_mapping', 'sequence_id']:
                    batch[k] = v.to(config['device'])
                    
            logits = model(batch)['logits']
            preds.append(logits.cpu().numpy())
            
            if model_no == 0: # only once
                offset_mapping = batch['offset_mapping']
                sequence_ids = batch['sequence_ids']
                offsets.append(offset_mapping.cpu().numpy())
                seq_ids.append(sequence_ids.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    if all_preds is None:
        all_preds = np.array(preds).astype(np.float32)
    else:
        all_preds += np.array(preds).astype(np.float32)
        
    torch.cuda.empty_cache()
    
    
all_preds /= len(paths)
all_preds = all_preds.squeeze()

offsets = np.concatenate(offsets, axis=0)
seq_ids = np.concatenate(seq_ids, axis=0)

print(all_preds.shape, offsets.shape, seq_ids.shape)

location_preds = get_location_predictions(all_preds, offsets, seq_ids, threshold=0.48, test=True)

test["location"] = location_preds
test[["id", "location"]].to_csv("submission.csv", index = False)