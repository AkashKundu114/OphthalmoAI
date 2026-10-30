import pandas as pd
import hashlib
from pathlib import Path
from collections import defaultdict

img_dir = Path('dataset/processed/images')
master_df = pd.read_csv('dataset/processed/master_patient_manifest.csv')

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

print('Computing MD5 for all 22,388 images...')
master_df['md5'] = master_df['image_id'].apply(get_md5)

total_unique_hashes = master_df['md5'].nunique()
print(f'Total records in master: {len(master_df)}')
print(f'Total unique MD5 hashes: {total_unique_hashes}')
print(f'Total duplicate image instances: {len(master_df) - total_unique_hashes}')

hash_to_classes = defaultdict(set)
hash_to_images = defaultdict(list)
for _, r in master_df.iterrows():
    h = r['md5']
    c = r['class']
    img = r['image_id']
    hash_to_classes[h].add(c)
    hash_to_images[h].append(img)

# Duplicates with conflicting classes
conflicting = {h: clss for h, clss in hash_to_classes.items() if len(clss) > 1}
print(f'Unique images that have CONFLICTING class labels: {len(conflicting)}')
conflicting_image_instances = sum(len(hash_to_images[h]) for h in conflicting)
print(f'Total image records with conflicting labels: {conflicting_image_instances}')

# Cross-split duplicate statistics
train_clean['md5'] = train_clean['image_id'].apply(get_md5)
val_clean['md5'] = val_clean['image_id'].apply(get_md5)
test_clean['md5'] = test_clean['image_id'].apply(get_md5)

tr_hashes = set(train_clean['md5'])
va_hashes = set(val_clean['md5'])
te_hashes = set(test_clean['md5'])

print('\nCross-split unique hash overlaps:')
print(f'Train & Test: {len(tr_hashes.intersection(te_hashes))}')
print(f'Train & Val:  {len(tr_hashes.intersection(va_hashes))}')
print(f'Val & Test:   {len(va_hashes.intersection(te_hashes))}')
print(f'Train & Val & Test: {len(tr_hashes.intersection(va_hashes).intersection(te_hashes))}')

# In test set, count duplicates having same class vs conflicting class with train
test_dup_same_class = 0
test_dup_conflict_class = 0
for _, r in test_clean.iterrows():
    h = r['md5']
    c = r['class']
    if h in tr_hashes:
        tr_classes = set(train_clean[train_clean['md5'] == h]['class'])
        if c in tr_classes:
            test_dup_same_class += 1
        else:
            test_dup_conflict_class += 1

print(f'\nIn Test Set (n=2249):')
print(f'  Duplicates identical in train with SAME class: {test_dup_same_class}')
print(f'  Duplicates identical in train with CONFLICTING class: {test_dup_conflict_class}')
print(f'  Total test images with byte-identical match in train: {test_dup_same_class + test_dup_conflict_class}')
