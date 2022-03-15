import ast
import joblib
import pandas as pd

features = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/features.csv')
train = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/train.csv')

pn = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/patient_notes.csv')

pn_dict = {}
for idx, row in pn.iterrows():
    pn_dict[row['pn_num']] = row['pn_history']

FOR_PP = {}

new_annotation = []
for case_id in features['case_num'].unique():
    
    all_pn_id = set(pn[pn['case_num']==case_id]['pn_num'].tolist())
    
    for feature_id in features[features['case_num']==case_id]['feature_num'].unique():
        
        # get all the pn_num that have already been annotated
        annotated_pn = set(train[train['feature_num']==feature_id]['pn_num'].tolist())
        
        # get all the pn_num that have NOT been annotated
        pn_to_annotate = all_pn_id-annotated_pn
        
        # get all current annotations
        # we will use them to find more annotations
        annotations = train[train['feature_num'] == feature_id]['annotation'].tolist()
        annotation_texts = set()
        
        for a in annotations:
            anns = eval(a)
            for at in anns:
                annotation_texts.add(at)
                FOR_PP[feature_id]  = FOR_PP.get(feature_id, []) + [at]                
                
        # annotate       
        for pn_id in pn_to_annotate:
            new_annotation_pn, new_location_pn = [], []
            pn_text = pn_dict[pn_id]
            for at in annotation_texts:
                start = pn_text.find(at)
                if start>=0:
                    new_annotation_pn.append(at)
                    new_location_pn.append(f'{start} {start+len(at)}')
                    
            if len(new_annotation_pn)>0:
                new_annotation.append((
                    f'{pn_id:04d}_{feature_id:03d}',
                    case_id,
                    pn_id,
                    feature_id,
                    new_annotation_pn,
                    new_location_pn
                ))


new_training_data = pd.DataFrame(new_annotation, columns=train.columns)
new_training_data['annotation'] = new_training_data['annotation'].astype(str)

new_training_data.to_csv('exact_match_train_data.csv', index=False)

df = pd.concat([train, new_training_data]).reset_index(drop=True)
df['annotation'] = df['annotation'].astype(str)

joblib.dump(FOR_PP, 'annots_for_pp.bin')