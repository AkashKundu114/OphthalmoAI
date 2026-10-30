import pandas as pd
import hashlib
from pathlib import Path

img_dir = Path('dataset/processed/images')

train_clean = pd.read_csv('dataset/processed/train_patient_clean.csv')
val_clean = pd.read_csv('dataset/processed/val_patient_clean.csv')
test_clean = pd.read_csv('dataset/processed/test_patient_clean.csv')

def get_md5(img_name):
    p = img_dir / img_name
    try:
        with open(p, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

test_clean['md5'] = test_clean['image_id'].apply(get_md5)
train_clean['md5'] = train_clean['image_id'].apply(get_md5)
val_clean['md5'] = val_clean['image_id'].apply(get_md5)

te_md5s = set(test_clean['md5'])
tr_md5s = set(train_clean['md5'])
va_md5s = set(val_clean['md5'])

common_te_tr = te_md5s.intersection(tr_md5s)
print(f'Total shared MD5 hashes between Test and Train: {len(common_te_tr)}')

sample_hashes = list(common_te_tr)[:10]
for h in sample_hashes:
    te_rows = test_clean[test_clean['md5'] == h]
    tr_rows = train_clean[train_clean['md5'] == h]
    print(f'\nHash: {h}')
    print('  Test:', te_rows[['image_id', 'patient_id', 'class', 'source_dataset']].to_dict(orient='records'))
    print('  Train:', tr_rows[['image_id', 'patient_id', 'class', 'source_dataset']].to_dict(orient='records'))

# Total count of test images affected
test_affected = test_clean[test_clean['md5'].isin(common_te_tr)]
print(f'\nTotal Test images that are identical duplicates in Train: {len(test_affected)} / {len(test_clean)} ({len(test_affected)/len(test_clean)*100:.2f}%)')
