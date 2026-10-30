import os
from pathlib import Path
from collections import Counter

raw_dir = Path('dataset/raw')

print('=== RAW CATEGORIES IN COLOR_FUNDUS_EYE ===')
cfe_dir = raw_dir / 'color_fundus_eye'
if cfe_dir.exists():
    folders = [p.name for p in cfe_dir.iterdir() if p.is_dir()]
    print('Folders in CFE:', len(folders))
    for f in sorted(folders):
        imgs = list((cfe_dir / f).glob('*.*'))
        print(f'  {f}: {len(imgs)} files')

print('\n=== RAW CATEGORIES IN APTOS_DR ===')
aptos_dir = raw_dir / 'aptos_dr'
if aptos_dir.exists():
    folders = [p.name for p in aptos_dir.iterdir() if p.is_dir()]
    print('Folders in APTOS:', len(folders))
    for f in sorted(folders):
        imgs = list((aptos_dir / f).glob('*.*'))
        print(f'  {f}: {len(imgs)} files')

print('\n=== RAW CATEGORIES IN PREPROCESSED_FUNDUS ===')
prep_dir = raw_dir / 'preprocessed_fundus'
if prep_dir.exists():
    folders = [p.name for p in prep_dir.iterdir() if p.is_dir()]
    print('Folders in PREP:', len(folders))
    for f in sorted(folders):
        imgs = list((prep_dir / f).glob('*.*'))
        print(f'  {f}: {len(imgs)} files')

print('\n=== RAW CATEGORIES IN ODIR5K ===')
odir_dir = raw_dir / 'odir5k'
if odir_dir.exists():
    folders = [p.name for p in odir_dir.iterdir() if p.is_dir()]
    print('Folders in ODIR:', len(folders))
    for f in sorted(folders):
        imgs = list((odir_dir / f).glob('*.*'))
        print(f'  {f}: {len(imgs)} files')
