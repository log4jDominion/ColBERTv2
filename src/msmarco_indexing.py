from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

import os

if __name__=='__main__':
    print(os.getcwd())
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            nbits=2,
            root="../experiments",
        )
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
        indexer.index(name="msmarco.nbits=2", collection="../resources/msmarco/collection.tsv", overwrite=True)

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            root="../experiments",
        )
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        queries = Queries("../resources/msmarco/queries/queries.dev.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.nbits=2.ranking.tsv")
