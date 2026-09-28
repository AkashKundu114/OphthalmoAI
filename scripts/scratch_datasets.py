import datasets

def check_dataset(repo_id):
    try:
        ds = datasets.load_dataset(repo_id, split="train", streaming=True)
        sample = next(iter(ds))
        print(f"\n{repo_id} columns:", list(sample.keys()))
        print(f"Sample: {sample}")
    except Exception as e:
        print(f"\nFailed {repo_id}: {e}")

check_dataset("amin-nejad/idrid-disease-grading")
check_dataset("ai4ophth/rim_one_dl_dataset")
check_dataset("OxAISH-AL-LLM/JSIEC")
check_dataset("Bingsu/cataract_classification")
check_dataset("jonathan-roberts1/OIA-ODIR")
