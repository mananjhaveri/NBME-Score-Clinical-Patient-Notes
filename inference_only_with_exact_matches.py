import pandas as pd
import numpy as np
import joblib
import itertools


def check_overlapping_preds(p1, p2):
    p1 = p1.split()
    p2 = p2.split()
    
    ret = {
        'accept': [],
        'reject': [],
    }
    
    if p1[1] < p2[0]:
        ret['accept'].extend([' '.join(p1), ' '.join(p2)])
    
    elif p2[1] < p1[0]:
        ret['accept'].extend([' '.join(p1), ' '.join(p2)])
    
    else:
        acc = [' '.join([min(p1[0], p2[0]), max(p1[1], p2[1])])]
        ret['accept'] += acc
        ret['reject'] += [i for i in [' '.join(p1), ' '.join(p2)] if i != acc[0]]
        
    return ret

annots = joblib.load('../input/nbme-exact-match-for-more-training-data/annots_for_pp.bin')

pn = pd.read_csv('../input/nbme-score-clinical-patient-notes/patient_notes.csv')
feats = pd.read_csv('../input/nbme-score-clinical-patient-notes/features.csv')
test = pd.read_csv('../input/nbme-score-clinical-patient-notes/test.csv')

df = test.merge(feats, on=['feature_num', 'case_num'], how='left')        .merge(pn, on=['pn_num', 'case_num'], how='left')

preds = []

for idx, row in df.iterrows():
    
    feat_id = row['feature_num']
    text = row['pn_history']
    
    pred_for_sample = []
    for annot in annots[feat_id]:
        start = text.find(annot)
        if start >= 0:
            pred_for_sample.append(f'{start} {start + len(annot)}')
    
    preds.append('; '.join(list(set(pred_for_sample))))

pp_for_pp_preds = []

for p in preds:
    
    p_list = p.split('; ')
    
    temp_preds = []
    rejected = []
    
    if len(p_list) > 1:
        comb = itertools.combinations(p_list, 2)
        for p1, p2 in comb:
            temp = check_overlapping_preds(p1, p2)
            
            for acc in temp['accept']:
                if acc not in rejected:
                    temp_preds += [acc]
    
            rejected += temp['reject']
        
        temp_preds = set(temp_preds) - set(rejected)
        pp_for_pp_preds.append('; '.join(list(temp_preds)))

    else:
        pp_for_pp_preds.append(p)

test['location'] = pp_for_pp_preds

sub = test[['id', 'location']]
sub.to_csv('submission.csv', index=False)