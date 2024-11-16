import datetime

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

collection = []
labels = []
index_name = 'sushi.colbert'
experiment_name = 'sushi.experiment'
checkpoint = 'colbert-ir/colbertv2.0'


def train_model(nbits, doc_maxlen):
    print(f"Indexing: {len(collection)} records")
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(nbits=nbits,
                               root=experiment_name)
        # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

    indexer.get_index()


def colbert_search(query):
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        config = ColBERTConfig(root=experiment_name)
        searcher = Searcher(index=index_name, config=config)

    # Find the top-5 passages for this query
    results = searcher.search(query, k=5)

    ranked_list = []

    for passage_id, passage_rank, passage_score in zip(*results):
        ranked_list.append(labels[passage_id])
        # print(f"\t{labels[passage_id]} \t\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

    return ranked_list


def train_colbert(training_data, training_labels):
    print(f"***********************Indexing starts at {datetime.datetime.now()}*************************")

    global collection
    collection = training_data

    global labels
    labels = training_labels

    train_model(2, 300)

    print(f'***********************Indexing ends at {datetime.datetime.now()}*************************')
