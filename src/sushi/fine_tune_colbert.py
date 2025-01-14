import os
import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sushi_main as main
from colbert import Indexer, Searcher, Trainer
from colbert.infra import Run, RunConfig, ColBERTConfig
from src.sushi.enums.env_vars import Vars

root = '/colbert_training'
experiment_name = 'colbert_fine_tuning'
index_name = 'sushi.fine.tuning.index'
checkpoint = "colbert-ir/colbertv2.0"
collection = None
lables = None


def find_most_dissimilar(strings, target_index):
    # Vectorize the strings using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(strings)

    # Compute cosine similarity between the target string and all others
    target_vector = tfidf_matrix[target_index]
    similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()

    # Set the similarity of the target string to itself to 1 (maximum similarity)
    similarities[target_index] = 1

    # Find the string with the lowest similarity score (most dissimilar)
    most_dissimilar_index = np.argmin(similarities)

    return most_dissimilar_index


def build_qrels(queries):
    qrels = []
    for i, query in enumerate(queries):
        negative_index = find_most_dissimilar(queries, i)
        qrels.append([i, i, negative_index])
    return qrels


def fine_tune_colbert():
    print("******************* Starting Fine Tuning *******************")
    base_url = os.getenv(Vars.RESOURCES.name)
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        #config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        config = ColBERTConfig(bsize=32, root='experiment_name')
        # trainer = Trainer(triples=base_url + '/msmarco/examples.json',
        #                   queries=base_url + '/msmarco/queries/queries.train.tsv',
        #                   collection=base_url + '/msmarco/collection.tsv',
        #                   config=config)

        trainer = Trainer(triples=base_url + '/sushi/SushiTriples.json',
                          queries=base_url + '/sushi/queries.tsv',
                          collection=base_url + '/sushi/collection.tsv',
                          config=config)

        trainer.train(checkpoint='colbert-ir/colbertv1.9')
        checkpoint_path = trainer.best_checkpoint_path()

        print(f"Saved checkpoint to {checkpoint_path}...")
        print("******************* Ending Fine Tuning *******************")
        global checkpoint
        checkpoint = checkpoint_path

        # indexer = Indexer(checkpoint=checkpoint_path, config=config)  # indexer.index(name=index_name, collection=collection, overwrite=True)


def colbert_query_search(query):
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        config = ColBERTConfig(root=root, )
        searcher = Searcher(index=index_name, config=config)
        results = searcher.search(query, k=1000)

        ranked_list = []

        for passage_id, passage_rank, passage_score in zip(*results):
            ranked_list.append(labels[
                                   passage_id])  # print(f"\t{labels[passage_id]} \t\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

        return ranked_list


def train_model(nbits, doc_maxlen):
    print(f"Indexing: {len(collection)} records")
    print(f"Training on checkpoint: {checkpoint}")
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(nbits=nbits, root=experiment_name, doc_maxlen=doc_maxlen)
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
        ranked_list.append(labels[
                               passage_id])  # print(f"\t{labels[passage_id]} \t\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

    return ranked_list


def train_colbert(training_data, training_labels):
    print(f"***********************Indexing starts at {datetime.datetime.now()}*************************")

    global collection
    collection = training_data

    global labels
    labels = training_labels

    train_model(2, 500)

    print(f'***********************Indexing ends at {datetime.datetime.now()}*************************')


if __name__ == '__main__':
    qrels = [[0, 101, 807], [1, 171, 521], [2, 1029, 182], [3, 864, 234], [4, 1197, 457], [5, 1240, 1251]]
    queries = {0: "Visit of Brazilian FORMIN", 1: "Goias Vice-Governorship", 2: "Presidential Elections",
               3: "Brazilian Coffee and Sugar Industry", 4: "Northeast Election Results",
               5: "IBAD Dissolved After Investigations"}
    labels = ['A9999907', 'A9999908', 'A9999909', 'A9999910', 'A9999911', 'A9999912']
    collection = [
        "Problems of Brazilian Rural Labor Union as Seen in Typical Northeastern Community LAB 3 Organizations & Conferences 1964 (Classified)  LIMITED OPPICIAL DSl Air Pouoh  DEPARTMENT OF 3TATB  INFO f RIO DE JANEIRO, BRASILIA, SALVADOR  AaConOen RECIFE i^ril 21^, 19614.  Problems of Brazilian Rural Labor Onion as Seen in  Typical Northeastern Qoiaraunity  A-138, April 23, 19614.  filBiARY AND INTRODUGTICM  Th9 drafting officer and Jose do Patrooinio OLiyEIRA,  director of the Recife office of 0 Olobo visited Vitoria de  Santo Antao, one of the foci of rural agitation in Pernambuco,  on April 17, 19614., The most ooiaplete tranquility appeared to  reign and not a single soldier was seen. Both the vicar of  the parish. Padre Renato da Cunha CWALGANTI, and the president  of the local syndicate of rural laborers, Manuel Alves de ARAUJO  Pilho, mphasizod the difficulties encountered by the rural  union which, before the revolution, had to fight on two fronts—  against the Ligas and against the majority of the landowners,  l^e main problem now facing the union, which presently enjoys  a monopoly situation, is its lack of trained leaders, facilities  and equipment, A suggestion is made that assistance, preferably  not tlrirough United States Ooverraaont agencies, be rendered to  unigns such as the Sindioato dos Trabalhadores Rurais de Santo  Antao.  SETTING  One of the centers of rural agitation in the Northeast,  the munlcipio of Vitoria de Santo Antao, had a population of  89,000 in 1900, It is typical of the humid zone (zona da mata)  in that its economy revolves around the growing a net processing  of sugar cane. Mill and plantation owners (uslneiros and senhores  de engenho) form its aristocracy while field hands (oamponeses)  form a teeming and miserable rural proletariat which, until ro- oent years, had evolved but little since the days of slavery,  LIMITED OFFICIAL USE  FExtonsje",
        "Basic Info About Caruaru POL 18 Pernambuco 1964 (Classified)  TELEGRAM  INCOMING Foreign Service of the  United States of America  LIMITED OPPICIAL USE  Classification Control:  Reed :Jan 2, 196it.  8:30 AM^y^^  PROM: RIO (/  NO : TOPAO 102, December 31 > ^1- PM  USITO 15.  Send basic info Caruaru for forwarding agency. Pictures other  illustrative material would be helpful securing sister city- affiliation.  BOERNER 6y  LIMITED OPPICIAL USE  REPRODUCTION FROM THIS COPY IS  Classification 'B'TEO UNLESS UNCLASSIFIED'r  FORM FS-412  MM POST ACTION COPY  GPO 89 20 57 ",
        "No Request for Overflight or Landing Clearance AV - Aviation (Civil) 1964 (Classified)  TELEGRAM  INCOMING Foreign Service of the  United States of America  AV 15«1  SECRET  Classification Control:  Reed: Mar 17, 1961+  9AM  PROM: PORT OP SPAIN  NO : 1, March 16, 10 AM  ACTION DEPT 328 INFO LONDON 55 RIO 11 CONAKRY 2 RECIPE 1  Reference: DEPCIRTEL 1673.  Sec External Affairs states still has n® request f®r overflight  or landing clearances Soviet aircraft.  GP-3  WOLLAM  GROUP 3  Downgraded at 12-year intervals,  not automatically declassified.  SECRET  FORM rs.4ij  M S5 Classification  POST ACTION COPY REPRODUCTION FROM THIS COPY IS  PROHIBITED UNLESS ""UNCLASSIFIED""  GPO 8920 57 ",
        "Armed Conflict on Farm Near Mari, Paraiba LAB 6 Labor Management Relations 1964 (Classified)  DECLASSIFIED  OUTGOING AmConGen RECIPE Foreign Service of the  United States of America  LIMITED OPPICIAL USE Chctrge: Classification Control: Date: Jan 16, 196ii. 3:30 PM ACTION: Amembassy RIO DE JANEIRO 206  Armed conflict yesterday on farm near Mari, Paraiba about  15 miles west Joao Pessoa resulted at least ten dead and dozen  wounded. Conflict followed attempts three local police and several  farm foremen and workers to investigate invasion farm by over two  hundred rural workers reportedly connected with Ligas Camponezaa.  Local leader Miri Liga, Antonio Galdino» among those reported killed.  Six members of group eight which investigating invasion were killed  and other two seriously injured in what practically amounted  massacre at hands large band invaders.  Governor Pedro Gondim has sent strong police reinforcements  into area and has cancelled plans attend housing conference in Chile.  Situation appears under control but exceedingly tense.  Consular Officer Kilday left this morning for Joao Pessoa to  get first hand account situation, particularly since ConGen receiving  increasing rumors possible large scale agitation among Paraiba rural  workers with leadership emanating from Pernambuco. Press reports  Juliao held meetings last Sunday in Sape near scene yesterday's  conflict.  LIMITED OFFICIAL USE REPRODUCTION FROM THIS COPY IS  PROHIBITED UNLESS ""UNCLASSIFIED"" Classification  FORM FS-413 S-1-5S GPO : 1961 O -61ZZ13 (69) ",
        "Northeast School Loan Program - USAID USAID-NE SECRET 1964-67  m LIMITED OFFICIAL USE  EMBASSY  At UNITED STATES OF AMERICA  Rio de Janeiro, Brazil  April 19, 1967  OF FICIAL - INF ORMAL  Grant H. Hilliker, Esquire  American Consul General  American Consulate General  Recife  Enclosed is a copy of a memorandum which 1 have sent to Bill Ellis  in response to the memorandum he furnished me on the subject of  the Northeast School Loan Program raised in your letter of April 5.  We appreciate having this problem brought to our attention and I am  happy to see that AID is sensitive to the complexities of the problem.  We attach a great deal of importance to education and of course it is  one of the most difficult areas in which to work because of Brazilian  sensitivities and the nationalistic attacks on our efforts to cooperate.  While solution of the education problem will not assure Brazil entry  into economic take-off/world power status, it is hard to envision the  country reaching that stage without solving it.  Sincerely,  7cnsuiate Genera! oi  United States of Amcr  APR 2 ? 1967  Philip Raine  Recife, Brazil Deputy Chief of Mission  Enclosure:  Copy of Memo to Mr. Ellis.  cc:USAlD:William Ellis  LIMITED OFFICIAL USE",
        "Specifications for the Construction of the American School in Recife EDU 9-5 - American School of Recife  0^  11 - I. ixION OF xHi; AFRICAN SCHCCL  TABLE Of COHxiaiTS  I) bwrn. ..I0I3  II) r.mcLiiios  1,.2 'G'vi^>--i  m) MCAVlilCSj ilLLlMCi LSfgtm  IV) CORCRETE  V) MASOSKT AMD PLABT^I  ¥1) WOOD PMI-tm  VII) ROOFS ARD FALSE C  VUl) GLASSES MD FOIWS  U) WATER lAMK  X) CI3l£li.IRGa  SI) PLOCF ijr.-^'-^ Vr CIAIGG  'iJDOD FLOORS MD fRULES 111)  nil)  m HARARE  Xfl) - L .  XVII) iffI«miC IMSxALLMiai  XVIII) ELECTRIC WSiALLAiiai  XIS) CLC...,.^;, -illD FITTIKGS  XX) PARKHJC PL.iCK. RS  ... GLOSEi'S"]

    main.set_env_vars()
    fine_tune_colbert()
