from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__ == "__main__":
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2,
            root="experiments",
        )
        indexer = Indexer(
            checkpoint="path/to/model_checkpoint",
            config=config,
        )
        indexer.index(
            name="uit_viquad_colbert_xml_base_maxsteps200000_bsize8_checkpoint20000",
            collection="int_collections.tsv",
            overwrite="resume",
        )
