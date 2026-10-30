import pandas as pd

train_clean = pd.read_csv('dataset/processed/train_patient_clean.csv')
val_clean = pd.read_csv('dataset/processed/val_patient_clean.csv')
test_clean = pd.read_csv('dataset/processed/test_patient_clean.csv')

def get_source(row):
    src = row.get('source_dataset')
    if pd.notnull(src):
        return src
    img = str(row.get('image_id'))
    if 'cfe' in img:
        return 'color_fundus_eye'
    elif 'odir' in img:
        return 'odir5k'
    elif 'aptos' in img:
        return 'aptos_dr'
    elif 'prep' in img:
        return 'preprocessed_fundus'
    else:
        return 'unknown'

train_clean['true_source'] = train_clean.apply(get_source, axis=1)
val_clean['true_source'] = val_clean.apply(get_source, axis=1)
test_clean['true_source'] = test_clean.apply(get_source, axis=1)

print('=== EXACT SOURCE DATASET BREAKDOWN ACROSS CLEAN SPLITS ===')
print('TRAIN (n={}):'.format(len(train_clean)))
print(train_clean['true_source'].value_counts())
print('\nVAL (n={}):'.format(len(val_clean)))
print(val_clean['true_source'].value_counts())
print('\nTEST (n={}):'.format(len(test_clean)))
print(test_clean['true_source'].value_counts())

# Master manifest
master_df = pd.read_csv('dataset/processed/master_patient_manifest.csv')
master_df['true_source'] = master_df.apply(get_source, axis=1)
print('\nMASTER MANIFEST (n={}):'.format(len(master_df)))
print(master_df['true_source'].value_counts())

# Unique patients per source in test_clean
print('\nUnique patients per source in TEST_CLEAN:')
for src, g in test_clean.groupby('true_source'):
    print(f'  {src}: {len(g)} images, {g["patient_id"].nunique()} unique patients')
