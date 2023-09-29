import sys

sys.path.append("../")
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__ == "__main__":
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            root="experiments",
        )
        searcher = Searcher(
            index="experiments/msmarco/indexes/uit_viquad_colbert_xml_base_maxsteps200000_bsize8_checkpoint20000",
            config=config,
        )
        queries = Queries("int_queries_test.tsv")
        ranking = searcher.search_all(queries, k=20)
        ranking.save("top20_ranking_test_xlm_200000_checkpoint20000.tsv")
