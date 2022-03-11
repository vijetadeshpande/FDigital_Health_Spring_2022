import json
import sys 
sys.path.insert(1, r'/Users/vijetadeshpande/Downloads/UMass Lowell - Courses/Spring 2022/Foundations in Digital Health/FDigital_Health_Spring_2022/Project_1')
import pandas as pd
import spacy
from quickumls import QuickUMLS
from UMedLS import UMedLS
from copy import deepcopy
import os
from tqdm import tqdm
import json

class AhsanSBDH(UMedLS):

    def __init__(self, dir_data = r'/Users/vijetadeshpande/Documents/GitHub/MIMIC-SBDH'):
        UMedLS.__init__(self)

        self.keyword = pd.read_csv(os.path.join(dir_data, 'MIMIC-SBDH-keywords.csv')).dropna()
        self.occurrence = pd.read_csv(os.path.join(dir_data, 'MIMIC-SBDH.csv')).dropna()

        return

    def get_sbdh_entities(self, text, mimic_row_identifier):

        """
        Here we make use of the previously published dataset, MIMIC-SBDH, that lists SBDH factors annotated for >7k
        MIMIC notes. The published data has two files,
            self.occurrence: binary matrix for presence of SBDH factors in annotated 7k notes
            self.keyword: location of the SBDH factors present in a particular note

        :param text: a medical note
        :param mimic_row_identifier: an identifier to search in the data published by Ahsan et.al.
        :return:
        """

        #
        try:
            kws = self.keyword.loc[self.keyword.loc[:, 'row_id'] == mimic_row_identifier, :]
        except:
            print('row_id not found in the keyword file')
            return []

        #
        entities = []
        for row in kws.index:
            start = int(kws.loc[row, 'start'])
            end = int(kws.loc[row, 'end'])
            ent_type = kws.loc[row, 'sbdh']
            ngram = text[start:end]

            #
            entity = {'start': start,
                      'end': end,
                      'ngram': ngram,
                      'term': None,
                      'cui': None,
                      'similarity': None,
                      'entity_type': ent_type,
                      }
            entities.append(deepcopy(entity))

        return entities

    def umls_search(self, term, table = 'MRCONSO'):

        res = self.search_term(table, term)

        return res

    def link_entities(self, text, entities, reference_set = []):

        """
        :param text: medical record text
        :param entities: entities for which we need to find cui
        :param reference_set: already found entities in 'text' via quickUMLS
        :return: entities (with added keys for cuis and other stuff)
        """

        # collect all terms identified by quickUMLS
        terms_found_by_quickumls, ngram_to_idx, idx_to_ngram = [], {}, {}
        for idx, ent in enumerate(reference_set):
            terms_found_by_quickumls.append(ent['ngram'])
            ngram_to_idx[ent['ngram']] = idx
            idx_to_ngram[idx] = ent['ngram']

        #
        remove_from_ref = []
        for idx, ent in enumerate(entities):
            if not ent['ngram'] in terms_found_by_quickumls:
                #TODO: current UMLS class code does not handle approximate search facility. It needs exact match.
                #res = self.umls_search('MRCONSO', ent['ngram'])
                #ent['cui'] = res['cui']
                #ent['term'] = None
                #ent['similarity'] = res['similarity']
                pass
            else:
                ent['cui'] = reference_set[ngram_to_idx[ent['ngram']]]['cui']
                ent['term'] = reference_set[ngram_to_idx[ent['ngram']]]['term']
                ent['similarity'] = reference_set[ngram_to_idx[ent['ngram']]]['similarity']

                # pop from the reference set to avoid repetition
                remove_from_ref.append(ngram_to_idx[ent['ngram']])

        #
        reference_set_out = []
        for idx, ent in enumerate(reference_set):
            if not idx in remove_from_ref:
                reference_set_out.append(ent)

        return entities, reference_set_out


class QuickUMLSPass(AhsanSBDH):

    def __init__(self, dir_quickumls = r'/Users/vijetadeshpande/Downloads/BioNLP Lab/Datasets/KG/UMLS/QuickUMLS'):
        AhsanSBDH.__init__(self)

        self.matcher = QuickUMLS(quickumls_fp = dir_quickumls)

        return

    def clean_umls_entities(self, ents):

        for ent in ents:
            ent['semtypes'] = list(ent['semtypes'])

        return ents

    def get_umls_entities(self, text):

        entities = self.matcher.match(text, best_match=True, ignore_syntax=False)

        return entities

    def get_top_match(self, matches):

        match_top = {}
        sim = 0
        for match in matches:
            if match['similarity'] == 1:
                return match

            if match['similarity'] > sim:
                match_top = match
                sim = match['similarity']

        return match_top

    def reduce_matches(self, ents_in):

        ents_out = []
        for ent in ents_in:
            top_match = self.get_top_match(ent)
            ents_out.append(top_match)

        return ents_out

    def get_entities(self, text, mimic_row_identifier):

        # get all type of entities in the text
        ents_ = self.get_umls_entities(text)
        ents_ = self.reduce_matches(ents_)
        ents_ = self.clean_umls_entities(ents_)
        ents_sbdh = self.get_sbdh_entities(text, mimic_row_identifier)

        # link all entities in the UMLS
        ents_sbdh, ents_ = self.link_entities(text, ents_sbdh, ents_)

        # merge both entities
        #entities = ents_ + ents_sbdh
        entities = {'sbdh': ents_sbdh, 'umls': ents_}

        return entities

    def annotate_document(self, text, mimic_row_identifier):

        # find entities and their location
        entities = self.get_entities(text, mimic_row_identifier)

        return {'text': text, 'entities': entities}

    def filter_data(self, df):

        # MIMIC-SBDH documents
        row_ids = self.occurrence.loc[:, 'row_id'].unique()
        row_idx = []
        for row_id in row_ids:
            idx = df.loc[df.loc[:, 'ROW_ID'] == row_id, :].index.values.tolist()
            row_idx += idx

        # filter mimic accordinly
        df = df.loc[row_idx, :].reset_index(drop=True)

        return df

    def annotate_documents(self, df, filter_data = True, annotated_data = {}):

        #
        if filter_data:
            df = self.filter_data(df)

        for row in tqdm(df.index):
            doc = df.loc[row, 'TEXT']
            row_id = int(df.loc[row, 'ROW_ID'])

            continue_ = True
            if str(row_id) in annotated_data:
                for doc_prev in annotated_data[str(row_id)]:
                    if doc_prev['text'] == doc:
                        continue_ = False
                        break

            if continue_:

                # annotation
                doc_annot = self.annotate_document(doc, row_id)

                # other features of the medical note
                doc_annot['CHARTDATE'] = str(df.loc[row, 'CHARTDATE'])
                doc_annot['SUBJECT_ID'] = int(df.loc[row, 'SUBJECT_ID'])
                doc_annot['HADM_ID'] = int(df.loc[row, 'HADM_ID'])

                #
                if not row_id in annotated_data:
                    annotated_data[str(row_id)] = [doc_annot]
                else:
                    annotated_data[str(row_id)].append(doc_annot)

                # write file every 500th iteration
                if (row%500) == 0:
                    print('Saving collected results')
                    with open('Data/mimic-sbdh-annotated.json', 'w') as f:
                        json.dump(annotated_data, f, indent=4, sort_keys=True)


        return annotated_data


#region Test the code

#data = pd.read_csv(r'/Users/vijetadeshpande/Downloads/BioNLP Lab/Datasets/EHR/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv', low_memory=False)
#print(data.columns)
#print(data.head())

#with open('Data/mimic-sbdh-annotated_read.json', 'r') as f:
#    annotated_docs = json.load(f)

#record = data.loc[data.loc[:, 'ROW_ID'] == 5, 'TEXT']
#text = record[0]
#qu_pass = QuickUMLSPass()
#data_anot = qu_pass.annotate_documents(data)#, annotated_data = annotated_docs)

#
#with open('Data/mimic-sbdh-annotated.json', 'w') as f:
#    json.dump(data_anot, f, indent = 4, sort_keys = True)

#endregion