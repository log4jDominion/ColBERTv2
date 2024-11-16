from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

import platform
import faiss


def get_base_url():
    if platform.system() == "Linux":
        return '/fs/clip-projects/archive_search/ColBERT_Code/ColBERTv2'
    elif platform.system() == "Darwin":
        return '/Users/shashank/projects/Colbert_Code/Colbert'
    else:
        return '/Users/shashank/projects/Colbert_Code/Colbert'


if __name__ == '__main__':

    base_url = get_base_url()

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2,
            root=base_url+"/experiments",
        )
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
        indexer.index(name="msmarco.nbits=2", collection=base_url+"/resources/msmarco/collection.dev.tsv", overwrite=True)

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            root=base_url+"/experiments",
        )
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        queries = Queries(base_url+"/resources/msmarco/queries/queries.dev.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.nbits=2.ranking.tsv")
