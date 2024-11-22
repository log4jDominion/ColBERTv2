from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher, Trainer

import platform
import faiss


def get_base_url():
    if platform.system() == "Linux":
        return '/fs/clip-projects/archive_search/ColBERT_Code/ColBERTv2/resources/msmarco'
    elif platform.system() == "Darwin":
        return '/Users/shashank/projects/Colbert_Code/Colbert'
    else:
        return '/Users/shashank/projects/Colbert_Code/Colbert'


if __name__ == '__main__':

    base_url = get_base_url()

    with Run().context(RunConfig(nranks=4)):
        triples = base_url+'/examples.json'  # `wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true` (26GB)
        queries = base_url+'/queries/queries.train.tsv'
        collection = base_url+'/collection.tsv'

        config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint='colbert-ir/colbertv1.9')
        checkpoint_path = trainer.best_checkpoint_path()

    with Run().context(RunConfig(nranks=3, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2,
            root=base_url+"/experiments",
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name="msmarco.nbits=2", collection=base_url+"/collection.dev.tsv", overwrite=True)


    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            root=base_url+"/experiments",
        )
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        queries = Queries(base_url+"/queries/queries.dev.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.nbits=2.ranking.tsv")
