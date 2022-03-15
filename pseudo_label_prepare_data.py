import numpy as np
import pandas as pd

from sklearn import model_selection

def create_folds(data):
    
    data['kfold'] = -1
    data['for_stratify'] = data['case_num'].astype(str) + '_' + data['feature_num'].astype(str)

    kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=541)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['for_stratify'])):
        data.loc[v_, 'kfold'] = f
    
    data.drop(['for_stratify'], axis=1, inplace=True)

    return data

pn = pd.read_csv('../input/nbme-score-clinical-patient-notes/patient_notes.csv')
feats = pd.read_csv('../input/nbme-score-clinical-patient-notes/features.csv')
train = pd.read_csv('../input/nbme-score-clinical-patient-notes/train.csv')

case_to_feats = {}

for case in feats['case_num'].unique().tolist():
    case_to_feats[case] = feats.query('case_num == @case')['feature_num'].tolist()


pn_x = pn.drop(['pn_history'], axis=1)
df_for_pseudo_labelling = []

for idx, row in pn_x.iterrows():
    case = row['case_num']
    pn_num = row['pn_num']
    
    for feat in case_to_feats[case]:
        df_for_pseudo_labelling.append((
            f'{pn_num:04d}_{feat:03d}',
            case,
            pn_num,
            feat,
        ))

df_for_pseudo_labelling = pd.DataFrame(df_for_pseudo_labelling, columns=['id', 'case_num', 'pn_num', 'feature_num'])
df_for_pseudo_labelling = create_folds(df_for_pseudo_labelling)
df_for_pseudo_labelling.to_csv('df_for_pseudo_labelling.csv', index=False)