import math
import sys
import os
import platform
import json
import pytrec_eval

from src.sushi.enums.env_vars import Vars
import sushi_dry_run_data as data_util
import train_colbert as colbert


def set_env_vars():
    os.environ[Vars.TXT.name] = "OCR"
    if platform.system() == "Linux":
        os.environ[Vars.JAVA_HOME.name] = "C:/Program Files/Java/jdk-22/"
        os.environ[Vars.PREFIX.name] = "/fs/clip-projects/archive_search/sushi/"
    elif platform.system() == "Darwin":
        os.environ[Vars.JAVA_HOME.name] = "C:/Program Files/Java/jdk-22/"
        os.environ[Vars.PREFIX.name] = "/Users/shashank/Research/sushi/"


def readExperimentControlFile():
    file_name = 'Ntcir18SushiDryRunExperimentControlFileV1.1.json'

    with open(os.getenv(Vars.PREFIX.name) + file_name) as ecfFile:
        ecf = json.load(ecfFile)
    return ecf


def get_search_fields():
    return ['title', 'ocr', 'folderlabel']


def eval_model():
    results = []
    i = 0

    for experimentSet in control_file['ExperimentSets']:
        training_dataset, labels = data_util.create_dry_run_data(experimentSet['TrainingDocuments'], search_fields)
        colbert.train_colbert(training_dataset, labels)
        topics = list(experimentSet['Topics'].keys())
        queries_list = []
        for j in range(len(topics)):
            results.append({})
            results[i]['Id'] = topics[j]
            query = experimentSet['Topics'][topics[j]]['TITLE']
            queries_list.append(query)
            rankedFolderList = colbert.colbert_search(query)
            results[i]['RankedList'] = rankedFolderList
            i += 1

    return results


def writeSearchResults(fileName, results, runName):
    with open(fileName, 'w') as f:
        for topic in results:
            for i in range(len(topic['RankedList'])):
                print(f'{topic["Id"]}\t{topic["RankedList"][i]}\t{i + 1}\t{1 / (i + 1):.4f}\t{runName}', file=f)
    f.close()


def createFolderToBoxMap(dir):
    boxMap = {}
    for box in os.listdir(dir):
        if not box.startswith("."):
            for folder in os.listdir(os.path.join(dir, box)):
                if folder in boxMap:
                    print(f'Duplicate folder {folder} in boxes {box} and {boxMap[folder]}')
                if not folder.startswith("."):
                    boxMap[folder] = box
    return boxMap


def makeBoxRun(folderRun):
    global prefix
    boxMap = createFolderToBoxMap(prefix + 'sushi-files/')
    boxRun = {}
    for topicId in folderRun:
        boxRun[topicId] = {}
        for folder in folderRun[topicId]:
            if boxMap[folder] not in boxRun[topicId]:
                boxRun[topicId][boxMap[folder]] = folderRun[topicId][folder]
    return boxRun


def stats(results, measure):
    sum = 0
    squaredev = 0
    n = len(results)
    for topic in results:
        sum += results[topic][measure]
    mean = sum / n
    for topic in results:
        squaredev += (results[topic][measure] - mean) ** 2
    variance = squaredev / (n - 1)
    conf = 1.96 * math.sqrt(variance) / math.sqrt(n)
    return mean, conf


def evaluateSearchResults(runFileName, folderQrelsFileName, boxQrelsFileName):
    #    print(pytrec_eval.supported_measures)
    measures = {'ndcg_cut', 'map', 'recip_rank', 'success'}  # Generic measures for configuring a pytrec_eval evaluator
    measureNames = {'ndcg_cut_5': 'NDCG@5', 'map': '   MAP', 'recip_rank': '   MRR',
                    'success_1': '   S@1'}  # Spedific measures for printing in pytrec_eval results

    with open(runFileName) as runFile, open(folderQrelsFileName) as folderQrelsFile, open(
            boxQrelsFileName) as boxQrelsFile:
        folderRun = {}
        for line in runFile:
            topicId, folderId, rank, score, runName = line.split('\t')
            if topicId not in folderRun:
                folderRun[topicId] = {}
            folderRun[topicId][folderId] = float(score)
        boxRun = makeBoxRun(folderRun)
        folderQrels = {}
        for line in folderQrelsFile:
            topicId, unused, folderId, relevanceLevel = line.split('\t')
            if topicId not in folderQrels:
                folderQrels[topicId] = {}
            folderQrels[topicId][folderId] = int(relevanceLevel.strip())  # this deletes the \n at end of line
        folderEvaluator = pytrec_eval.RelevanceEvaluator(folderQrels, measures)
        folderTopicResults = folderEvaluator.evaluate(
            folderRun)  # replace run with folderQrels to see perfect evaluation measures

        boxQrels = {}
        for line in boxQrelsFile:
            topicId, unused, folderId, relevanceLevel = line.split('\t')
            if topicId not in boxQrels:
                boxQrels[topicId] = {}
            if folderId in boxQrels[topicId]:
                boxQrels[topicId][folderId] = max(boxQrels[topicId][folderId],
                                                  int(relevanceLevel.strip()))  # strip() deletes the \n at end of line
            else:
                boxQrels[topicId][folderId] = int(relevanceLevel.strip())
        boxEvaluator = pytrec_eval.RelevanceEvaluator(boxQrels, measures)
        boxTopicResults = boxEvaluator.evaluate(boxRun)  # replace run with qrels to see perfect evaluation measures

        pm = '\u00B1'
        print(f'          Folder          Box')
        for measure in measureNames.keys():
            folderMean, folderConf = stats(folderTopicResults, measure)
            boxMean, boxConf = stats(boxTopicResults, measure)
            print(f'{measureNames[measure]}: {folderMean:.3f}{pm}{folderConf:.2f}    {boxMean:.3f}{pm}{boxConf:.2f}')


if __name__ == '__main__':
    set_env_vars()
    control_file = readExperimentControlFile()
    search_fields = get_search_fields()

    prefix = os.getenv(Vars.PREFIX.name)

    results = eval_model()
    writeSearchResults(prefix + 'Ntcir18SushiDryRunResultsV1.1.tsv', results, 'Baseline-0')
    evaluateSearchResults(prefix + 'Ntcir18SushiDryRunResultsV1.1.tsv',
                          prefix + 'Ntcir18SushiDryRunFolderQrelsV1.1.tsv',
                          prefix + 'Ntcir18SushiDryRunBoxQrelsV1.1.tsv')
