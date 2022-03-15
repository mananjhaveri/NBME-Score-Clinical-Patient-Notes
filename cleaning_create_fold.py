import numpy as np
import pandas as pd
import ast
from sklearn import model_selection

SEED = 541

def create_folds(data, num_folds):
    
    data['kfold'] = -1
    data['for_stratify'] = data['case_num'].astype(str) + '_' + data['feature_num'].astype(str)

    kf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['for_stratify'])):
        data.loc[v_, 'kfold'] = f
    
    data.drop(['for_stratify'], axis=1, inplace=True)

    return data

def create_train_df(debug=False):
    ROOT = '../input/nbme-score-clinical-patient-notes'

    feats = pd.read_csv(f"{ROOT}/features.csv")
    feats.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    
    notes = pd.read_csv(f"{ROOT}/patient_notes.csv")
    train = pd.read_csv(f"{ROOT}/train.csv")
    extra_data = pd.read_csv(
        '../input/nbme-exact-match-for-more-training-data/exact_match_train_data.csv'
    )

    
    train['annotation'] = train['annotation'].apply(ast.literal_eval)
    train['location'] = train['location'].apply(ast.literal_eval)
    
    extra_data['annotation'] = extra_data['annotation'].apply(ast.literal_eval)
    extra_data['location'] = extra_data['location'].apply(ast.literal_eval)
    
    train.loc[338, 'annotation'] = ast.literal_eval('[["father heart attack"]]')
    train.loc[338, 'location'] = ast.literal_eval('[["764 783"]]')

    train.loc[621, 'annotation'] = ast.literal_eval('[["for the last 2-3 months"]]')
    train.loc[621, 'location'] = ast.literal_eval('[["77 100"]]')

    train.loc[655, 'annotation'] = ast.literal_eval('[["no heat intolerance"], ["no cold intolerance"]]')
    train.loc[655, 'location'] = ast.literal_eval('[["285 292;301 312"], ["285 287;296 312"]]')

    train.loc[1262, 'annotation'] = ast.literal_eval('[["mother thyroid problem"]]')
    train.loc[1262, 'location'] = ast.literal_eval('[["551 557;565 580"]]')

    train.loc[1265, 'annotation'] = ast.literal_eval('[[\'felt like he was going to "pass out"\']]')
    train.loc[1265, 'location'] = ast.literal_eval('[["131 135;181 212"]]')

    train.loc[1396, 'annotation'] = ast.literal_eval('[["stool , with no blood"]]')
    train.loc[1396, 'location'] = ast.literal_eval('[["259 280"]]')

    train.loc[1591, 'annotation'] = ast.literal_eval('[["diarrhoe non blooody"]]')
    train.loc[1591, 'location'] = ast.literal_eval('[["176 184;201 212"]]')

    train.loc[1615, 'annotation'] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
    train.loc[1615, 'location'] = ast.literal_eval('[["249 257;271 288"]]')

    train.loc[1664, 'annotation'] = ast.literal_eval('[["no vaginal discharge"]]')
    train.loc[1664, 'location'] = ast.literal_eval('[["822 824;907 924"]]')

    train.loc[1714, 'annotation'] = ast.literal_eval('[["started about 8-10 hours ago"]]')
    train.loc[1714, 'location'] = ast.literal_eval('[["101 129"]]')

    train.loc[1929, 'annotation'] = ast.literal_eval('[["no blood in the stool"]]')
    train.loc[1929, 'location'] = ast.literal_eval('[["531 539;549 561"]]')

    train.loc[2134, 'annotation'] = ast.literal_eval('[["last sexually active 9 months ago"]]')
    train.loc[2134, 'location'] = ast.literal_eval('[["540 560;581 593"]]')

    train.loc[2191, 'annotation'] = ast.literal_eval('[["right lower quadrant pain"]]')
    train.loc[2191, 'location'] = ast.literal_eval('[["32 57"]]')

    train.loc[2553, 'annotation'] = ast.literal_eval('[["diarrhoea no blood"]]')
    train.loc[2553, 'location'] = ast.literal_eval('[["308 317;376 384"]]')

    train.loc[3124, 'annotation'] = ast.literal_eval('[["sweating"]]')
    train.loc[3124, 'location'] = ast.literal_eval('[["549 557"]]')

    train.loc[3858, 'annotation'] = ast.literal_eval('[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]')
    train.loc[3858, 'location'] = ast.literal_eval('[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]')

    train.loc[4373, 'annotation'] = ast.literal_eval('[["for 2 months"]]')
    train.loc[4373, 'location'] = ast.literal_eval('[["33 45"]]')

    train.loc[4763, 'annotation'] = ast.literal_eval('[["35 year old"]]')
    train.loc[4763, 'location'] = ast.literal_eval('[["5 16"]]')

    train.loc[4782, 'annotation'] = ast.literal_eval('[["darker brown stools"]]')
    train.loc[4782, 'location'] = ast.literal_eval('[["175 194"]]')

    train.loc[4908, 'annotation'] = ast.literal_eval('[["uncle with peptic ulcer"]]')
    train.loc[4908, 'location'] = ast.literal_eval('[["700 723"]]')

    train.loc[6016, 'annotation'] = ast.literal_eval('[["difficulty falling asleep"]]')
    train.loc[6016, 'location'] = ast.literal_eval('[["225 250"]]')

    train.loc[6192, 'annotation'] = ast.literal_eval('[["helps to take care of aging mother and in-laws"]]')
    train.loc[6192, 'location'] = ast.literal_eval('[["197 218;236 260"]]')

    train.loc[6380, 'annotation'] = ast.literal_eval('[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]')
    train.loc[6380, 'location'] = ast.literal_eval('[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]')

    train.loc[6562, 'annotation'] = ast.literal_eval('[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]')
    train.loc[6562, 'location'] = ast.literal_eval('[["290 320;327 337"], ["290 320;342 358"]]')

    train.loc[6862, 'annotation'] = ast.literal_eval('[["stressor taking care of many sick family members"]]')
    train.loc[6862, 'location'] = ast.literal_eval('[["288 296;324 363"]]')

    train.loc[7022, 'annotation'] = ast.literal_eval('[["heart started racing and felt numbness for the 1st time in her finger tips"]]')
    train.loc[7022, 'location'] = ast.literal_eval('[["108 182"]]')

    train.loc[7422, 'annotation'] = ast.literal_eval('[["first started 5 yrs"]]')
    train.loc[7422, 'location'] = ast.literal_eval('[["102 121"]]')

    train.loc[8876, 'annotation'] = ast.literal_eval('[["No shortness of breath"]]')
    train.loc[8876, 'location'] = ast.literal_eval('[["481 483;533 552"]]')

    train.loc[9027, 'annotation'] = ast.literal_eval('[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]')
    train.loc[9027, 'location'] = ast.literal_eval('[["92 102"], ["123 164"]]')

    train.loc[9938, 'annotation'] = ast.literal_eval('[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]')
    train.loc[9938, 'location'] = ast.literal_eval('[["89 117"], ["122 138"], ["368 402"]]')

    train.loc[9973, 'annotation'] = ast.literal_eval('[["gaining 10-15 lbs"]]')
    train.loc[9973, 'location'] = ast.literal_eval('[["344 361"]]')

    train.loc[10513, 'annotation'] = ast.literal_eval('[["weight gain"], ["gain of 10-16lbs"]]')
    train.loc[10513, 'location'] = ast.literal_eval('[["600 611"], ["607 623"]]')

    train.loc[11551, 'annotation'] = ast.literal_eval('[["seeing her son knows are not real"]]')
    train.loc[11551, 'location'] = ast.literal_eval('[["386 400;443 461"]]')

    train.loc[11677, 'annotation'] = ast.literal_eval('[["saw him once in the kitchen after he died"]]')
    train.loc[11677, 'location'] = ast.literal_eval('[["160 201"]]')

    train.loc[12124, 'annotation'] = ast.literal_eval('[["tried Ambien but it didnt work"]]')
    train.loc[12124, 'location'] = ast.literal_eval('[["325 337;349 366"]]')

    train.loc[12279, 'annotation'] = ast.literal_eval('[["heard what she described as a party later than evening these things did not actually happen"]]')
    train.loc[12279, 'location'] = ast.literal_eval('[["405 459;488 524"]]')

    train.loc[12289, 'annotation'] = ast.literal_eval('[["experienced seeing her son at the kitchen table these things did not actually happen"]]')
    train.loc[12289, 'location'] = ast.literal_eval('[["353 400;488 524"]]')

    train.loc[13238, 'annotation'] = ast.literal_eval('[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
    train.loc[13238, 'location'] = ast.literal_eval('[["293 307"], ["321 331"]]')

    train.loc[13297, 'annotation'] = ast.literal_eval('[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]')
    train.loc[13297, 'location'] = ast.literal_eval('[["182 221"], ["182 213;225 234"]]')

    train.loc[13299, 'annotation'] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
    train.loc[13299, 'location'] = ast.literal_eval('[["79 88"], ["409 418"]]')

    train.loc[13845, 'annotation'] = ast.literal_eval('[["headache global"], ["headache throughout her head"]]')
    train.loc[13845, 'location'] = ast.literal_eval('[["86 94;230 236"], ["86 94;237 256"]]')

    train.loc[14083, 'annotation'] = ast.literal_eval('[["headache generalized in her head"]]')
    train.loc[14083, 'location'] = ast.literal_eval('[["56 64;156 179"]]')

    merged = train.merge(notes, how = "left")
    merged = merged.merge(feats, how = "left")
    merged = create_folds(merged, 5)
    
    merged_extra = extra_data.merge(notes, how='left')
    merged_extra = merged_extra.merge(feats, how='left')
    merged_extra = create_folds(merged_extra, 10)
    merged_extra['kfold'] = (merged_extra['kfold'] + 1) * -1
    
    merged = pd.concat([merged, merged_extra]).reset_index(drop=True)
    
    def process_feature_text(text):
        return text.replace("-OR-", ";-").replace("-", " ")
    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    
    merged["feature_text"] = merged["feature_text"].apply(lambda x: x.lower())
    merged["pn_history"] = merged["pn_history"].apply(lambda x: x.lower())
    
    return merged

train_df = create_train_df()
display(train_df.head())


train_df.to_csv('train_with_external_data.csv', index=False)

train_df.kfold.value_counts()

train = train_df.query('kfold >= 0').reset_index(drop=True)
train.to_csv('train.csv', index=False)
train.kfold.value_counts()

extra_data = train_df.query('kfold < 0').reset_index(drop=True)
extra_data.to_csv('extra_data.csv', index=False)
extra_data.kfold.value_counts()

