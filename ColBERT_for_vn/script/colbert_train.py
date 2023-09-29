import sys

sys.path.append("../")
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
            triples="colbert_vn_data/only_int_triples.json",
            queries="colbert_vn_data/int_queries_train.tsv",
            collection="colbert_vn_data/int_collections.tsv",
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint="xlm-roberta-base")

        print(f"Saved checkpoint to {checkpoint_path}...")
