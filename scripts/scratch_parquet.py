import pandas as pd

try:
    df1 = pd.read_parquet("hf://datasets/amin-nejad/idrid-disease-grading/data/train-00000-of-00001.parquet")
    print("IDRID cols:", df1.columns)
except Exception as e:
    print(e)
