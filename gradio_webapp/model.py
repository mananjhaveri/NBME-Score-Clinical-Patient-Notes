import gc

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange

import torch
from torch import nn
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig


config = dict(
    # basic
    seed = 3407,
    num_jobs=1,
    num_labels=2,
    
    # model info
    tokenizer_path = 'roberta-large', # 'allenai/biomed_roberta_base',
    model_checkpoint = 'roberta-large', # 'allenai/biomed_roberta_base', 
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # training paramters
    max_length = 512,
    batch_size=16,
    
    # for this notebook
    debug = False,
)


def create_sample_test():
    feats = pd.read_csv(f"../input/nbme-score-clinical-patient-notes/features.csv")
    feats.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    
    notes = pd.read_csv(f"../input/nbme-score-clinical-patient-notes/patient_notes.csv")
    test = pd.read_csv(f"../input/nbme-score-clinical-patient-notes/test.csv")

    merged = test.merge(notes, how = "left")
    merged = merged.merge(feats, how = "left")

    def process_feature_text(text):
        return text.replace("-OR-", ";-").replace("-", " ")
    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    
    return merged.sample(1).reset_index(drop=True)

class NBMETestData(torch.utils.data.Dataset):
    def __init__(self, feature_text, pn_history, tokenizer):
        self.feature_text = feature_text
        self.pn_history = pn_history
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.feature_text)
    
    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.feature_text[idx],
            self.pn_history[idx],
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

# class NBMEModel(nn.Module):
#     def __init__(self, num_labels=1, path=None):
#         super().__init__()
        
#         layer_norm_eps: float = 1e-6
        
#         self.path = path
#         self.num_labels = num_labels
        
#         self.transformer = transformers.AutoModel.from_pretrained(config['model_checkpoint'])
#         self.dropout = nn.Dropout(0.2)
#         self.output = nn.Linear(768, 1)
        
#         if self.path is not None:
#             self.load_state_dict(torch.load(self.path)['model'])

#     def forward(self, data):
        
#         ids = data['input_ids']
#         mask = data['attention_mask']
#         try:
#             target = data['targets']
#         except:
#             target = None

#         transformer_out = self.transformer(ids, mask)
#         sequence_output = transformer_out[0]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.output(sequence_output)

#         ret = {
#             "logits": torch.sigmoid(logits),
#         }
        
#         if target is not None:
#             loss = self.get_loss(logits, target)
#             ret['loss'] = loss
#             ret['targets'] = target

#         return ret

        
#     def get_optimizer(self, learning_rate, weigth_decay):
#         optimizer = torch.optim.AdamW(
#             self.parameters(), 
#             lr=learning_rate, 
#             weight_decay=weigth_decay,
#         )
#         if self.path is not None:
#             optimizer.load_state_dict(torch.load(self.path)['optimizer'])
        
#         return optimizer
            
#     def get_scheduler(self, optimizer, num_warmup_steps, num_training_steps):
#         scheduler = transformers.get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=num_warmup_steps,
#             num_training_steps=num_training_steps,
#         )
#         if self.path is not None:
#             scheduler.load_state_dict(torch.load(self.path)['scheduler'])
            
#         return scheduler
    
#     def get_loss(self, output, target):
#         loss_fn = nn.BCEWithLogitsLoss(reduction="none")
#         loss = loss_fn(output.view(-1, 1), target.view(-1, 1))
#         loss = torch.masked_select(loss, target.view(-1, 1) != -100).mean()
#         return loss


class NBMEModel(nn.Module):
    def __init__(self, num_labels=2, path=None):
        super().__init__()
        
        layer_norm_eps: float = 1e-6
        
        self.path = path
        self.num_labels = num_labels
        self.transformer = transformers.AutoModel.from_pretrained(config['model_checkpoint'])
        self.dropout = nn.Dropout(0.1)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(1024, 1)
        
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


def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        start_idx = None
        current_preds = []
        for p, o, s_id in zip(pred, offsets, seq_ids):
            if s_id is None or s_id == 0:
                continue
            if p > 0.5:
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



def predict_location_preds(tokenizer, model, feature_text, pn_history, pn_history_lower):

    test_ds = NBMETestData(feature_text, pn_history_lower, tokenizer)
    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=config['batch_size'], 
        pin_memory=True, 
        shuffle=False, 
        drop_last=False
    )

    all_preds = None
    offsets = []
    seq_ids = []

    preds = []

    with torch.no_grad():
        for batch in tqdm(test_dl):

            for k, v in batch.items():
                if k not in  ['offset_mapping', 'sequence_id']:
                    batch[k] = v.to(config['device'])

            logits = model(batch)['logits']
            preds.append(logits.cpu().numpy())

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

    all_preds = all_preds.squeeze()

    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)

    # print(all_preds.shape, offsets.shape, seq_ids.shape)

    location_preds = get_location_predictions([all_preds], offsets, seq_ids, test=False)[0]
    
    x = []
    
    for location in location_preds:
        x.append(pn_history[0][location[0]: location[1]])
    
    return location_preds, ', '.join(x)

def get_predictions(feature_text, pn_history):
    feature_text = feature_text.lower().replace("-OR-", ";-").replace("-", " ")
    pn_history_lower = pn_history.lower()
    
    location_preds, pred_string = predict_location_preds(tokenizer, model, [feature_text], [pn_history], [pn_history_lower])
    
    if pred_string == "":
        pred_string = 'Feature not present!'
    else:
        pred_string = 'Feature is present!' + '\nText Span - ' + pred_string
    
    return pred_string

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
path = 'model_large_pseudo_label.pth'

model = NBMEModel().to(config['device'])
model.load_state_dict(
    torch.load(
        path, 
        map_location=torch.device(config['device'])
    )
)
model.eval()