import datetime
from ragatouille import RAGPretrainedModel
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter

collection = []
labels = []
index_name = 'sushi.colbert'
experiment_name = 'sushi.experiment'
checkpoint = 'colbert-ir/colbertv2.0'

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


def train_model(nbits, doc_maxlen):
    RAG.index(
        collection=collection,
        document_metadatas=[{"entity": "sushi", "source": "ocr"}],
        index_name=index_name,
        max_document_length=doc_maxlen,
        split_documents=True
    )


def colbert_search(query):
    results = RAG.search(query=query, k=1000)

    # results = RAG.rerank(query=query, documents=collection, k=1000)
    #
    ranked_list = []
    #
    # for result in results:
    #     score = result['score']
    #     rank = result['rank']
    #     passage_id = result['result_index']
    #     print(f'Score: {score}, Rank: {rank}, Passage_ID: {passage_id}')
    #     ranked_list.append(labels[passage_id])

    for passage_id, rank, score in zip(*results):
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
