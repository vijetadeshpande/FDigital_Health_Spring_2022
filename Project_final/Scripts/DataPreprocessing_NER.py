import json
import random

import torch.nn as nn
import medspacy
import re
from tqdm import tqdm
from copy import deepcopy
import spacy
from medspacy.ner import TargetRule
import string

class PreprocessData:
    def __init__(
            self,
            datafile: str
    ):
        # read data
        with open(datafile, 'r') as f:
            self.data = json.load(f)

        # medspacy model
        #nlp = spacy.load("en_core_web_sm", disable={"ner"})
        nlp = medspacy.load()
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
        target_rules = [
            TargetRule("atrial fibrillation", "PROBLEM"),
            TargetRule("atrial fibrillation", "PROBLEM", pattern=[{"LOWER": "afib"}]),
            TargetRule("pneumonia", "PROBLEM"),
            TargetRule("Type II Diabetes Mellitus", "PROBLEM",
                       pattern=[
                           {"LOWER": "type"},
                           {"LOWER": {"IN": ["2", "ii", "two"]}},
                           {"LOWER": {"IN": ["dm", "diabetes"]}},
                           {"LOWER": "mellitus", "OP": "?"}
                       ]),
            TargetRule("warfarin", "MEDICATION")
        ]
        target_matcher.add(target_rules)
        self.nlp = nlp

        return

    def fill_entity_type(
            self,
            text: str,
            entities: dict,
            label_name: str = 'SBDH',
            entity_type: str = 'entity_type'
    ):
        # sort entities by their occurrence
        starts = []
        ents = {}
        for ent in entities:
            start = ent['start']
            # @TODO: there are instances where in the list of entities, same span has two entries
            # I have checked few of those and multiple entries were identical. But I haven't checked thouroughly.
            # Following line: we assume that all multiple occurrences are same and ignore all after the first
            if not start in ents:
                ents[start] = ent
                starts.append(start)
        starts = sorted(starts)

        #
        text = deepcopy(text)
        offset = 0
        prev_end = -1
        for start in starts:
            end = ents[start]['end']
            word = ents[start]['ngram']

            # find start of the ngram after 'start'
            word_starts = [text[start+offset:].find(ents[start]['ngram'])]#[m.start() for m in re.finditer(ents[start]['ngram'], text[start+offset:])]
            word_start = start + offset + word_starts[0]
            word_end = word_start + len(word)

            #
            if not word_starts[0] == 0:
                print('\ngadbad')
                print(word_starts[0])
                continue

            # label to replace the word text with
            #label = '[' + label_name + ':%s' % (ents[start][entity_type]) + ']'
            label = '[ENT-%d-%d]' % (ents[start]['start'], ents[start]['end'])


            # update the text
            if prev_end == word_start:
                # there are some instances where two separate entities are present together in the text
                # and there's no space between them
                text = text[:word_start] + ' ' + label + text[word_end:]
            else:
                # update the text
                text = text[:word_start] + label + text[word_end:]
            prev_end = word_end

            # update offset (next iteration, the match should be at starts + offset)
            offset += len(label) - len(word)

        # @TODO: following line is a very hard coded trick, need to be careful about it
        # this is to deal with joint entity mentions
        text = text.replace(']/[ENT-', ']/ [ENT-')
        text = text.replace('][ENT-', '] [ENT-')
        text = text.replace(',[ENT-', ', [ENT-')
        text = text.replace('].[ENT-', ']. [ENT-')
        text = text.replace(']:[', ']: [')


        return text

    def label_notes(
            self
    ):

        #
        data = deepcopy(self.data)

        #
        labeled_notes = {}
        for mimic_idx in tqdm(data):
            labeled_notes[mimic_idx] = {}

            # features of data
            text = data[mimic_idx][0]['text']
            ents_sbdh = data[mimic_idx][0]['entities']['sbdh']
            ents_umls = data[mimic_idx][0]['entities']['umls']

            # create copies of raw text with different labels
            """
            labeled_notes[mimic_idx]['text_unlabeled'] = text
            labeled_notes[mimic_idx]['text_labeled_SBDH'] = self.fill_entity_type(
                text=text,
                entities=ents_sbdh,
                label_name='SBDH',
                entity_type='entity_type',
            )
            labeled_notes[mimic_idx]['text_labeled_UMLS'] = self.fill_entity_type(
                text=text,
                entities=ents_sbdh,
                label_name='UMLS',
                entity_type='cui',
            )
            """
            labeled_notes[mimic_idx]['text_labeled_SBDH'] = self.fill_entity_type(
                text=text,
                entities=ents_sbdh,
                label_name='SBDH',
                entity_type='entity_type',
            )

        return labeled_notes

    def into_paragraph(
            self,
            text: str
    ):

        # split based on '\n'
        note_ = text.replace(':\n', ': ')
        candidates = note_.split('\n')

        #
        paragraphs = []
        rejected = []
        idx = 0
        while idx < len(candidates):
            par = ''
            while not candidates[idx] == '':
                par += candidates[idx] + ' '
                idx += 1
                if idx >= len(candidates):
                    break

            if not par == '':
                paragraphs.append(par)
            idx += 1

        return paragraphs

    def into_sentences(
            self,
            text: str
    ):
        # first we split into paragraphs
        paragraphs = self.into_paragraph(
            text=text
        )

        # iterate over all paragraphs and split it into sentences
        sent_idx = -1
        sentences = []
        positive_idx = []
        for par in paragraphs:
            par = self.nlp(par)
            for sent in par.sents:
                sent_idx += 1
                sent_text = sent.text
                sent_text = ' '.join(sent_text.split())
                sentences.append(sent_text)
                #if '[SBDH:' in sent_text or '[UMLS:' in sent_text:
                if '[ENT:' in sent_text:
                    positive_idx.append(sent_idx)

        return sentences, positive_idx

    def is_correct_pair(
            self,
            array_input: str,
            array_label: str
    ):
        if (array_input[:5] == array_label[:5]) and (array_input[-5:] == array_label[-5:]):
            return True

        if (array_input[:5] == array_label[:5]) or (array_input[-5:] == array_label[-5:]):
            if ('[SBDH:' in array_label) or ('[UMLS:' in array_label):
                return True

        print('arrays are not matching')
        print(array_input)
        print(array_label)

        return False

    def get_token_category(
            self,
            token,
    ):
        #
        gadbad = False

        # find start
        i = token.find('[ENT-')
        if i > 0:
            gadbad = True

        # find end
        i_end = len(token)
        while i_end > i:
            i_end -= 1
            if token[i_end] == ']':
                if i_end < len(token) - 2: # -2 because ',' is present in many cases at the end
                    gadbad = True
                break

        #
        #if gadbad:
        #    print('Word containing entity tag also includes other characters')
        #    print('Word is: %s'%token)

        # update token based on start and end
        token_ = token[i:i_end]
        token_ = token_.strip()
        token_ = token_.strip(string.punctuation)

        #
        category = token_.split(':')[-1]
        start = token_.split('-')[1]
        end = token_.split('-')[-1]

        return int(start), int(end)

    def reshape_entity_dictionary(
            self,
            entities: dict
    ):
        ents = {}
        for ent in entities:
            start = ent['start']
            end = ent['end']
            ents[(start, end)] = ent

        return ents

    def get_data_instances(
            self,
            input_seq: list,
            entities: dict,
            label_sbdh: list = None,
            label_umls: list = None
    ):
        data_instances = []

        # restructure entities
        ents = self.reshape_entity_dictionary(entities=entities)

        # chedk if the length is same:
        #if not len(input_seq) == len(label_sbdh) == len(label_umls):
        #    return data_instances

        for idx, x in enumerate(input_seq):
            #y_sbdh = label_sbdh[idx]
            #y_umls = label_umls[idx]

            # check if strings are matching
            #if not (self.is_correct_pair(x, y_sbdh) and (self.is_correct_pair(x, y_umls))):
            #    continue

            # chedk if the length is same:
            #if not len(x) == len(y_sbdh) == len(y_umls):
            #    continue

            #
            words = x.split(' ')
            labels_sbdh = ['O'] * len(words)
            labels_umls = ['O'] * len(words)

            #
            """
            if '[SBDH:' in y_sbdh:
                for i, token in enumerate(y_sbdh.split(' ')):
                    if '[SBDH' in token:
                        cat_sbdh = self.get_token_category(token)
                        cat_umls = self.get_token_category(y_umls.split(' ')[i])

                        # replace values
                        labels_sbdh[i] = cat_sbdh
                        labels_umls[i] = cat_umls

                        # randomly print x and labels to check
                        if random.random() > 0.7:
                            print('\nPrinting word and related labels:')
                            print('word: %s'% words[i])
                            print('sdbh: %s'% token)
            """
            if '[ENT-' in x:
                for i, token in enumerate(words):
                    if ('[ENT-' in token) and (']' in token):

                        try:
                            start, end = self.get_token_category(token)
                        except:
                            print('\nNot able to split the word into start and end reference')
                            print('word is: %s' % token)
                            continue

                        try:
                            cat_sbdh = ents[(start, end)]['entity_type']
                            cat_umls = ents[(start, end)]['cui']
                            n_gram = ents[(start, end)]['ngram']
                        except:
                            print('Very weird situation, entity is present in the text but not found in dict')
                            print('word is: %s' % token)
                            continue

                        # replace values
                        labels_sbdh[i] = cat_sbdh
                        labels_umls[i] = cat_umls
                        words[i] = token.replace('[ENT-%d-%d]' % (start, end), n_gram)

                        # randomly print x and labels to check
                        #if random.random() > 0.7:
                        #    print('\nPrinting word and related labels:')
                        #    print('word: %s'% words[i])
                        #    print('sdbh: %s'% token)

            # create instance
            instance = {
                'words': words,
                'labels_sbdh': labels_sbdh,
                'labels_umls': labels_umls,
            }
            data_instances.append(deepcopy(instance))

        return data_instances

    def create_labeled_sequences(
            self
    ):
        # first we will replace entity spans in the notes with a reference ID
        labeled_notes = self.label_notes()

        # now we have medical notes with ref IDs
        labeled_sequences = {}
        data_instances = {}
        for note_idx, mimic_idx in tqdm(enumerate(labeled_notes)):
            labeled_sequences[mimic_idx] = {}

            """
            x_, _ = self.into_sentences(
                text=labeled_notes[mimic_idx]['text_unlabeled']
            )

            y_sbdh, y_sbdh_pos = self.into_sentences(
                text=labeled_notes[mimic_idx]['text_labeled_SBDH']
            )

            y_umls, y_umls_pos = self.into_sentences(
                text=labeled_notes[mimic_idx]['text_labeled_UMLS']
            )
            
            # check
            if not len(x_) == len(y_sbdh) == len(y_umls):
                print('\nThere is a difference between number of sentences in x and y')
                print('x length is: %d' % len(x_))
                print('y length is: %d' % len(y_sbdh))
                print('y length is: %d' % len(y_umls))

            #
            labeled_sequences[mimic_idx] = {
                'sentences': deepcopy(x_),
                'sentences_SBDH': deepcopy(y_sbdh),
                'sentences_SBDH_pos': deepcopy(y_sbdh_pos),
                'sentences_UMLS': deepcopy(y_umls),
                'sentences_UMLS_pos': deepcopy(y_umls_pos)
            }
            """

            #
            sentences, pos_sent_idx = self.into_sentences(
                text=labeled_notes[mimic_idx]['text_labeled_SBDH']
            )

            #
            note_instances = self.get_data_instances(
                input_seq=sentences,
                entities=self.data[mimic_idx][0]['entities']['sbdh'],
            )

            # create data instances from extracted sentences
            data_instances[mimic_idx] = {
                'sequences': note_instances,
                'sbdh_positive_index': pos_sent_idx
            }

            #
            if note_idx%500 == 0:
                with open('test_required_form.json', 'w') as f:
                    json.dump(data_instances, fp=f, indent=4)

        # asve after the loop
        with open('test_required_form.json', 'w') as f:
            json.dump(data_instances, fp=f, indent=4)

        return data_instances


# test
#tt = PreprocessData(
#    datafile=r'/Users/vijetadeshpande/Downloads/UMass Lowell - Courses/Spring 2022/Foundations in Digital Health/FDigital_Health_Spring_2022/Project_final/Data/mimic-sbdh-annotated.json'
#)

#
#data_ = tt.create_labeled_sequences()
