# ColBERT for Vietnamese
This directory is for training ColBERT model for Vietnamese

# Requirements
Clone the official [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) and following the instructions in the [Installation](https://github.com/stanford-futuredata/ColBERT#installation) section.

# Data
Download the processed data colbert_vn_data.zip. The data includes the following files.
- only_int_triples.json
- int_queries_train.tsv
- int_collections.tsv

# Training
Use the script `script/colbert_train.py` for training ColBERT Vietnamese. Modify the following part of the script to the your appropriate paths. 

```python
import sys

sys.path.append("/path/to/ColBERT") # <--- Path to the cloned ColBERT repository from https://github.com/stanford-futuredata/ColBERT
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__ == "__main__":
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            bsize=8,
            root="./experiments",
            maxsteps=200_000,
        )
        trainer = Trainer(
            triples="colbert_vn_data/only_int_triples.json", # <--- Path to the downloaded data
            queries="colbert_vn_data/int_queries_train.tsv", # <--- Path to the downloaded data
            collection="colbert_vn_data/int_collections.tsv", # <--- Path to the downloaded data
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint="xlm-roberta-base")

        print(f"Saved checkpoint to {checkpoint_path}...")
```
