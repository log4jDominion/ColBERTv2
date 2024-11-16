import os
import re

import PyPDF2
import pandas as pd

from src.sushi.enums.env_vars import Vars


def translateNaraFolderLabel(naraLabel, sncExpansion, sushiFile, sushiFolder):
    if naraLabel != 'nan':
        start = naraLabel.find('(')  # Strip part markings
        if start != -1:
            naraLabel = naraLabel[:start]
        naraLabel = naraLabel.replace('BRAZ-A0', 'BRAZ-A 0')  # Fix formatting error
        naraLabel = naraLabel.replace('BRAZ-E0', 'BRAZ-E 0')  # Fix formatting error
        naraLabelElements = naraLabel.split()
        if len(naraLabelElements) in [3, 4]:
            if len(naraLabelElements) == 3:
                naraSnc = naraLabelElements[0]
            else:
                naraSnc = ' '.join(naraLabelElements[0:2])
            naraCountryCode = naraLabelElements[-2]
            naraDate = naraLabelElements[-1]
            #                print(f'parsed {naraLabel} to {naraSnc} // {naraCountryCode} // {naraDate}')
            if naraSnc in sncExpansion['SNC'].tolist():
                label1965 = str(sncExpansion.loc[sncExpansion['SNC'] == naraSnc, 1965].iloc[0])
                label1963 = str(sncExpansion.loc[sncExpansion['SNC'] == naraSnc, 1963].iloc[0])
                if label1965 != 'nan':
                    label = label1965
                elif label1963 != 'nan':
                    label = label1963
                else:
                    print(f'Unable to translate {naraSnc} for file {sushiFile} in folder {sushiFolder}')
                    label = naraSnc
            else:
                print(f'No expansion for {naraSnc}')
                label = naraSnc
        else:
            print(f"NARA Folder Title doesn't have four parts: {naraLabel}")
            label = 'Bad NARA Folder Title'
    return label, naraCountryCode, naraDate


def translateBrownFolderLabel(brownLabel, sncExpansion, sushiFile, sushiFolder):
    label = ''
    cleanLabel = brownLabel.replace('_', ' ')
    datePattern = re.compile(r'(^.*)([\s\-])(1?\d[\-\/][123]?\d[\-\/]?[67]\d)(.*$)')
    match = datePattern.search(cleanLabel)
    if match:
        before = match.group(1).strip()
        date = match.group(3).strip()
        if date[-2] in ['6', '7'] and date[-3] not in ['-', '/', ' ']:
            date = date[0:-2] + '-' + date[-2:]
        #        else:
        #            print(f'Date: {date} Date[-2]: {date[-2]}')
        after = match.group(4).strip()
    else:
        before = cleanLabel
        after = ''

    catPattern = re.compile(r'(^[A-Z][A-Za-z]{0,4})(.*)$')
    match = catPattern.search(before)
    if match:  # and before!='Unknown' and 'Untitled' not in before:
        category = match.group(1).strip()
        subcats = match.group(2).strip().strip('-').strip()
    if not match or len(category) > 4:
        category = ''
        subcats = ''
        label = before

    snc1Pattern = re.compile(r'(^\d\d?)(.*)')
    match = snc1Pattern.search(subcats)
    if match:
        level1 = match.group(1).strip()
        remainder = match.group(2).strip()
    else:
        level1 = ''
        remainder = subcats

    snc2Pattern = re.compile(r'(\-)(\d\d?)(.*)')
    match = snc2Pattern.search(remainder)
    if match:
        level2 = match.group(2).strip()
        remainder = match.group(3).strip() + ' ' + after
        remainder = remainder.strip()
    else:
        level2 = ''
        remainder = remainder + ' ' + after
        remainder = remainder.strip().strip('-').strip()

    if category != '':
        if level1 != '':
            if level2 != '':
                brownSnc = category.upper() + ' ' + level1 + '-' + level2
            else:
                brownSnc = category.upper() + ' ' + level1
        else:
            brownSnc = category.upper()
    else:
        brownSnc = 'Unknown'

    if brownSnc in sncExpansion['SNC'].tolist():
        label1965 = str(sncExpansion.loc[sncExpansion['SNC'] == brownSnc, 1965].iloc[0])
        label1963 = str(sncExpansion.loc[sncExpansion['SNC'] == brownSnc, 1963].iloc[0])
        if label1965 != 'nan':
            label2 = label1965
        elif label1963 != 'nan':
            label2 = label1963
        else:
            print(f'Unable to translate {brownSnc} for file {sushiFile} in folder {sushiFolder}')
            label2 = brownSnc
    else:
        print(f'No expansion for {brownSnc}')
        label2 = 'Unknown'

    if label == '':
        if label2 == 'Unknown':
            label = remainder
        else:
            label = label2

    if len(cleanLabel) < 20:
        return label
    else:
        return cleanLabel


def create_trainingSet(trainingDocs):
    noShortOcr = False  # Set to true if you want to replace OCR text that is nearly empty with the document title
    fileMetadata = None
    sncExpansion = None
    prefix = os.getenv(Vars.PREFIX.name)
    trainingSet = []

    # Read the Sushi Medadata and SNC excel files
    try:
        xls = pd.ExcelFile(prefix + 'SubtaskACollectionMetadataV1.1.xlsx')
        fileMetadata = xls.parse(xls.sheet_names[0])
        xls = pd.ExcelFile(prefix + 'SncTranslationV1.2.xlsx')
        sncExpansion = xls.parse(xls.sheet_names[0])
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit(-1)

    # Build the data structure that Terrier will index (list of dicts, one per indexed item)
    for trainingDoc in trainingDocs:

        # Read the box/folder/file directory structure
        sushiFile = trainingDoc[
                    -10:]  # This extracts the file name and ignores the box and folder labels which we will get from the medatada
        file = sushiFile
        folder = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Sushi Folder'].iloc[0])
        box = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Sushi Box'].iloc[0])

        # Construct the best available folder label (either by SNC lookup or by using the folder label text)
        naraLabel = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'NARA Folder Name'].iloc[0])
        brownLabel = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Brown Folder Name'].iloc[0])
        if naraLabel != 'nan':
            label = translateNaraFolderLabel(naraLabel, sncExpansion, file, folder)
        else:
            if brownLabel != 'nan':
                label = translateBrownFolderLabel(brownLabel, sncExpansion, file, folder)
            else:
                print(f'Missing both NARA and Brown folder labels for file {file} in folder {folder}')
                label = 'No NARA or Brown Folder Title'

        brownTitle = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Brown Title'].iloc[0])
        naraTitle = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'NARA Title'].iloc[0])
        if brownTitle != 'nan':
            title = brownTitle
        else:
            start = naraTitle.find('Concerning')
            if start != -1:
                naraTitle = naraTitle[start + 11:]
            end1 = naraTitle.rfind(':')
            end2 = naraTitle.rfind('(')
            end = min(end1, end2)
            if end != -1:
                naraTitle = naraTitle[:end]
            title = naraTitle

        ocr = ''
        summary = ''
        if os.getenv(Vars.TXT.name).__contains__('GPT'):
            f = open(prefix + 'summary/prompt-1/' + box + '/' + folder + '/' + file.replace('.pdf', '.txt'), 'rt')
            summary = f.read()
        elif os.getenv(Vars.TXT.name).__contains__('OCR'):
            # Extract OCR text from the PDF file
            f = open(prefix + 'sushi-files/' + box + '/' + folder + '/' + file, 'rb')
            reader = PyPDF2.PdfReader(f)
            pages = len(reader.pages)
            maxPages = 1  # Increase this number if you want to index more of the OCR text
            fulltext = ''
            for i in range(min(pages, maxPages)):
                page = reader.pages[i]
                text = page.extract_text().replace('\n', ' ')
                fulltext = fulltext + text
            ocr = fulltext

            # Optionally replace any hopelessly short OCR with the document title
            if noShortOcr and len(ocr) < 5:
                print(f'Replaced OCR: //{ocr}// with Title //{title}//')
                ocr = title

        text = summary + ' ' + ocr
        trainingSet.append(
            {'docno': file, 'folder': folder, 'box': box, 'title': title, 'ocr': text, 'folderlabel': label})

    return trainingSet


def extract_label_training_dataset(experiment_set, search_fields):
    training_set = create_trainingSet(experiment_set)

    training_data = {key: [] for key in search_fields}

    for data in training_set:
        for search_field in search_fields:
            training_data[search_field].append(data[search_field])

    merged_text_list = [list(values) for values in zip(*training_data.values())]

    merged_text = []
    for lst in merged_text_list:
        complete_str = Vars.CLS.value
        for i, elem in enumerate(lst):
            if type(elem) is tuple:
                elem = ''.join(elem)
            complete_str += elem
            if i < len(lst) - 1:
                complete_str += Vars.SEP.value
        complete_str += Vars.CLS.value
        merged_text.append(complete_str)

    label_text = [data[Vars.FOLDER.value] for data in training_set]

    return merged_text, label_text


def create_dry_run_data(experiment_set, search_fields):
    return extract_label_training_dataset(experiment_set, search_fields)
