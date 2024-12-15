# MakeSubTaskATestCollection.py
# Version 1.2 as of 11/29/2024
# Douglas W. Oard

# This program was originally designed to create the Experiment Control File (ECF) for the SUSHI Dry Run
# It is also capable of generating larger ECFs that use title metadata as queries
# It can also read assessor-created queries, for use in the official run
# The main method is SetupECF, which prepares the data structures that are needed to write an ECF

import os
import json
import pandas as pd
import random
import math


def sortLongest(my_dict):
    # Given a set of folders for a box, sort them in decreasing order of the number of documents they contain
    dict_lengths = {key: len(value) for key, value in my_dict.items()}
    sorted_keys = sorted(dict_lengths, key=lambda k: dict_lengths[k], reverse=True)
    sorted_dict = {key: my_dict[key] for key in sorted_keys}
    return sorted_dict


def getSushiFiles(dir):
    fullCollection = {}
    print("Starting walk of", dir)
    for box in os.listdir(dir):
        if not box.startswith('.'):
            print(f'Reading SUSHI collection box {box}')
            fullCollection[box] = {}
            for folder in os.listdir(os.path.join(dir, box)):
                if not folder.startswith('.'):
                    fullCollection[box][folder] = []
                    #            print(f'Read box {box}, folder {folder}')
                    for file in os.listdir(os.path.join(dir, box, folder)):
                        #                print(f'Read file {os.path.join(dir,box,folder,file)}')
                        fullCollection[box][folder].append(file)
    for box in fullCollection:
        fullCollection[box] = sortLongest(fullCollection[box])
    return fullCollection


def readQueries(file_path):
    # The queries are document titles, from the best available title metadata.  Some duplicates are present.
    # These queries are stored in excel, with one sheet per query length (for lengths 2, 3, 4, and 5)
    try:
        xls = pd.ExcelFile(file_path)
        sheet_data = {}
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if "SELECT" in df.columns and "Query Candidate" in df.columns and "Sushi Box" in df.columns and "Sushi Folder" in df.columns and "Sushi File" in df.columns:
                sheet_data[sheet_name] = df[["SELECT", "Query Candidate", "Sushi Box", "Sushi Folder", "Sushi File"]]
            else:
                print(f"Specified columns not found in Dry Run Queries spreadsheet tab '{sheet_name}'.")
        return sheet_data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


def readOfficialTopics(file_path):
    # The official topics were created by annotators, with fields for title, description and narrative
    try:
        xls = pd.ExcelFile(file_path)
        sheet_data = {}
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if "Title" in df.columns and "Description" in df.columns and "Narrative" in df.columns:
                sheet_data[sheet_name] = df[["Title", "Description", "Narrative"]]
            else:
                print(f"Specified columns not found in Official Topics spreadsheet tab '{sheet_name}'.")
        return sheet_data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


def createDryRunTopics(queryFiles, selected):
    # SUSHI topics have title, description and narrative fields, but we only have titles.  So we copy that three times.
    titles = []
    for queryFile in queryFiles:
        titles.append(selected.loc[selected['Sushi File'] == queryFile, 'Query Candidate'].iloc[0])
    topicSet = []
    for i in range(len(titles)):
        topicSet.append({})
        topicSet[i]['TITLE'] = titles[i]
        topicSet[i]['DESCRIPTION'] = titles[i]
        topicSet[i]['NARRATIVE'] = titles[i]
    return topicSet


def selectUniformTraining(queryFiles, fullCollection, docsPerBox, shuffleFolders=False):
    # This selects the training documents for a set of queries (which must already have been selected)
    # It will select the same number of documents training for every box (which is simple, but not realistic)
    # For each box, round robin selection is used to choose training docs from as many different folders as possible
    trainingSet = []
    trainingFiles = []
    max = 300  # To simplify the data structure, we consider only the first 300 folders per box
    for box in fullCollection:
        # First we need to decide how many documents to take from each folder
        # There may be fewer folders in the box than the number of documents we want.  The inner loop handles that.
        folderDocs = [0] * max
        total = 0
        if shuffleFolders:  # Don't do this for the Dry Run collection!
            # print(f'Shuffling {len(fullCollection[box])} folders')
            items = list(fullCollection[box].items())
            random.shuffle(items)
            fullCollection[box] = dict(items)
        for j in range(docsPerBox):
            for i in range(min(len(fullCollection[box]), docsPerBox, max)):
                # Be sure that a folder has enough docs for the docs already selected and for the query (in case the query is in this box)
                n = len(fullCollection[box][list(fullCollection[box])[i]])
                if n > folderDocs[i] + 1 and total < docsPerBox:
                    folderDocs[i] += 1
                    total += 1
        #        print(folderDocs)
        # Now we randomly select that number of documents from each folder.
        # We need to be careful not to choose one of our query documents as a training document.
        i = 0
        for folder in fullCollection[box]:
            for j in range(folderDocs[i]):
                candidate = random.choice(fullCollection[box][folder])
                while candidate in queryFiles or candidate in trainingFiles:
                    if candidate in queryFiles:
                        print(f'Initial random selection of {candidate} was found in Query Files and replaced.')
                    else:
                        print(f'Initial random selection of {candidate} was found in Training Files and replaced.')
                        print(f'Folder Length: {len(fullCollection[box][folder])}')
                    candidate = random.choice(fullCollection[box][folder])
                trainingFiles.append(candidate)
                trainingSet.append(box + '/' + folder + '/' + candidate)
            #                print(f'{box}/{folder}/{candidate}')
            i += 1
        trainingSet.sort()
    return trainingSet


def setupEcf(queryTables, fullCollection, setSize=100, maxSets=99, condition='Dry Run'):
    queries = []  # Despite the name, this holds the full set of SUSHI File IDs
    queryFiles = []  # This is a list of lists, one per training set, each of which contains SUSHI file IDs
    topicSets = []  # This is a list of lists, on per training set, each of which
    trainingSets = []  # This is a list of lists, one per training set, each of whcih contains SUSHI file IDs
    random.seed(100)  # This ensures that the "random" selections are the same every time the program is run
    if queryTables:
        # We start by concatenating the queries of different lengths
        df = pd.concat([*queryTables.values()])
        if condition == 'Official':
            if len(df.index) >= setSize * maxSets:
                for i in range(maxSets):
                    queryFiles.append([])
                    topicSets.append([])
                    for j in range(setSize):
                        row = df.iloc[i * setSize + j].to_dict()
                        topicSets[i].append({})
                        topicSets[i][j]["TITLE"] = row['Title']
                        topicSets[i][j]["DESCRIPTION"] = row['Narrative']
                        topicSets[i][j]["NARRATIVE"] = row['Narrative']
        else:
            if condition == 'Dry Run':
                # Assessors manually selected 50 Dry Run queries of each length, as indicated by the SELECT field
                selected = df.query('@pd.notna(SELECT)')
            else:  # In this case, condition must be 'Testing'
                # If we want more than those 50*4=200 queries, we just dedupe the full query set
                # Note that this skews the length distribution a bit, since the number of queries differs for each length
                # It also results in somewhat lower query quality, since they have not been checked for reasonableness
                selected = df.drop_duplicates(subset=['Query Candidate'])
            uniqueQueries = selected['Query Candidate'].unique()
            for query in uniqueQueries:
                doc = selected.loc[selected['Query Candidate'] == query, 'Sushi File'].iloc[0]
                queries.append(doc)
            print(f'Total of {len(queries)} unique queries are available.')
            # Now we pick the set of queries that will be used with each training set.
            # Because we ultimately want topics, not just queries, we also set up the topic set for each training set
            random.shuffle(queries)
            for i in range(min(math.trunc(len(queries) / setSize), maxSets)):
                queryFiles.append(queries[i * setSize:(i + 1) * setSize])
                topicSets.append(createDryRunTopics(queryFiles[i], selected))

        # Now we are ready to generate the Training Set
        for i in range(len(topicSets)):
            if condition == 'Dry Run':
                trainingSets.append(selectUniformTraining(queryFiles[i], fullCollection, 5, shuffleFolders=False))
            else:
                trainingSets.append(selectUniformTraining(queryFiles[i], fullCollection, 5, shuffleFolders=True))
            print(f'Topic Set {i} with length {len(topicSets[i])}: {topicSets[i]}')
            print(f'Training Set {i} with length {len(trainingSets[i])}: {trainingSets[i]}')
            if condition != 'Official':
                print(f'Query Files Set {i} with length {len(queryFiles[i])}: {queryFiles[i]}')
                # Double check to make sure we didn't somehow screw up and get a query file in the training set
                for queryFile in queryFiles[i]:
                    for path in trainingSets[i]:
                        if queryFile in path:
                            print(f'Error: Query File {queryFile} found in Training Set {path}')
    else:
        print('Unable to read Excel file for queries')
        exit(-1)
    return topicSets, trainingSets


def writeJson(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def writeEcf(fileName, experimentName, trainingSets, topicSets, topicPrefix, firstTopicNumber):
    ecf = {}
    ecf['ExperimentName'] = experimentName
    ecf['ExperimentSets'] = []
    if len(trainingSets) != len(topicSets):
        print(f'Mismatch between {trainingSets} Training Sets and {topicSets} Topic Sets; Aborted')
        exit(-1)
    for set in range(len(trainingSets)):
        ecf['ExperimentSets'].append({})
        ecf['ExperimentSets'][set]['TrainingDocuments'] = trainingSets[set]
        ecf['ExperimentSets'][set]['Topics'] = {}
        for topic in range(len(topicSets[set])):
            topicId = topicPrefix + '-' + '{:05}'.format(firstTopicNumber + topic)
            ecf['ExperimentSets'][set]['Topics'][topicId] = topicSets[set][topic]
            topicSets[set][topic]['ID'] = topicId
        firstTopicNumber += len(topicSets[set])
    print(f'Writing to file {fileName}')
    writeJson(ecf, fileName)
    return topicSets


def setupQrels(queryTables, topicSets):
    df = pd.concat([*queryTables.values()])
    for topicSet in topicSets:
        for topic in topicSet:
            selected = df[df['Query Candidate'] == topic['TITLE']]
            topic['qrelFiles'] = selected['Sushi File'].drop_duplicates().to_list()
            topic['qrelFolders'] = selected['Sushi Folder'].drop_duplicates().to_list()
            topic['qrelBoxes'] = selected['Sushi Box'].drop_duplicates().to_list()
            print(f'Topic: {topic["ID"]}, Query: {topic["TITLE"]}, Sushi Folders: {topic["qrelFolders"]}')
    return topicSets


def writeQrels(folderQrelsFileName, boxQrelsFileName, topicSets):
    f = open(folderQrelsFileName, 'w')
    g = open(boxQrelsFileName, 'w')
    for topicSet in topicSets:
        for topic in topicSet:
            for relFolder in topic['qrelFolders']:
                print(f'{topic["ID"]}\t0\t{relFolder}\t3', file=f)
            for relBox in topic['qrelBoxes']:
                print(f'{topic["ID"]}\t0\t{relBox}\t3', file=g)
    f.close()
    g.close()


if __name__ == '__main__':
    condition = 'Testing'  # Choices are Dry Run, Official, Testing
    prefix = '/fs/clip-projects/archive_search/sushi/'
    fullCollection = getSushiFiles(prefix + 'sushi-files/')
    queryTables = readQueries(prefix + 'titlequeries.xlsx')
    officialTopicTables = readOfficialTopics(prefix + 'SushiSubtaskADevTopicsNov29.xlsx')
    if condition == 'Dry Run':
        topicSets, trainingSets = setupEcf(queryTables, fullCollection, setSize=100, maxSets=2, condition=condition)
        topicSets = writeEcf('/Users/shashank/Research/sushi/Ntcir18SushiDryRunExperimentControlFileV1.1.json',
                             'NTCIR-18 SUSHI Dry Run v1.1, 200 Queries, Uniform Sample, 5 Documents Per Box',
                             trainingSets, topicSets, 'T18DryRun', 1)
        topicSets = setupQrels(queryTables, topicSets)
        writeQrels('/Users/shashank/Research/sushi/Ntcir18SushiDryRunFolderQrelsV1.1.tsv',
                   '/Users/shashank/Research/sushi/Ntcir18SushiDryRunBoxQrelsV1.1.tsv', topicSets)
        print('Subtask A Dry Run Test Collection and Qrels Built')
    elif condition == 'Official':
        topicSets, trainingSets = setupEcf(officialTopicTables, fullCollection, setSize=5, maxSets=2,
                                           condition=condition)
        print(f'topicSets: {topicSets}')
        topicSets = writeEcf('/Users/shashank/Research/sushi/NtcirSushiOfficialExperimentControlFileV1.0.json',
                             'Debugging Version of Official ECF with 10 Topics', trainingSets, topicSets,
                             'T18Eval', 1)
        print('Subtask A Official Test Collection Built (without Qrels)')
    else:
        topicSets, trainingSets = setupEcf(queryTables, fullCollection, setSize=100, maxSets=100, condition=condition)
        topicSets = writeEcf(prefix + 'ecftest.json', 'Dry Run-Like Test Set of Selected Size', trainingSets, topicSets,
                             'TEST', 1)
        topicSets = setupQrels(queryTables, topicSets)
        writeQrels(prefix + 'folderQrelsTest.tsv', prefix + 'boxQrelsTest.tsv', topicSets)
        print('Subtask A Training Test Collection and Qrels Built')
