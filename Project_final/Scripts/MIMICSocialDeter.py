import torch
import random
from torch.utils.data import Dataset
import scipy.sparse
from pdb import set_trace as breakpoint
from collections import Counter
from random import shuffle
import json
from transformers import AutoTokenizer

class MIMICSocialDeter(Dataset):
    """
    This class takes list of data instances in a specific split (say train) and
    returns tokenized form of the data when __getitemm__ is called.
    """

    def __init__(
            self,
            list_data: list,
            tokenizer,
            label2id: dict,
            id2label: dict,
            max_length: int = 512
    ):

        # attributes
        self.data = list_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.id2label = id2label

        return

    def __len__(self):
        """Returns total length or total data instances in the input data"""
        return self.data.__len__()

    def __getitem__(self, item) -> dict:
        """This is the main function of the class. Returns tokenized form of the called item"""

        # get the 'item' index data instance
        example = self.data[item]

        # tokenize the example
        tokenized = self.tokenizer(
            example['words'],
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True, # this is key argument for token classification task
            #add_prefix_space=True
        )

        # check if there are multiple sequences due to truncation


        # extend the labels if words are split into multiple tokens
        labels_sbdh, labels_umls = [], []
        word_ids = tokenized.word_ids(batch_index=0)
        prev_id = None
        for word_id in word_ids:

            if word_id is None:
                labels_sbdh.append(-100)
                labels_umls.append(-100)
            else:
                label_sbdh = str(example['labels_sbdh'][word_id])
                label_umls = str(example['labels_umls'][word_id])

                if label_sbdh == 'O':
                    label_sbdh = label_sbdh
                    label_umls = label_umls
                else:
                    if word_id != prev_id:
                        label_sbdh = label_sbdh + '_b'
                        label_umls = label_umls + '_b'
                    elif word_id == prev_id:
                        label_sbdh = label_sbdh + '_i'
                        label_umls = label_umls + '_i'
                    else:
                        print('Labeling logic is not correct')

                #
                labels_sbdh.append(self.label2id[label_sbdh])
                labels_umls.append(self.label2id[label_sbdh])

            # update prev_id
            prev_id = word_id

        # check
        assert len(tokenized['input_ids']) == len(labels_sbdh) == len(labels_umls)

        #
        tokenized['labels_sbdh'] = labels_sbdh
        tokenized['labels_umls'] = labels_umls

        return tokenized


# test the code
filepath = r'/Users/vijetadeshpande/Downloads/UMass Lowell - Courses/Spring 2022/Foundations in Digital Health/FDigital_Health_Spring_2022/Project_final/Scripts/preprocessed_data_version_2.json'
with open(filepath, 'r') as f:
    data = json.load(f)

#
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

trial = MIMICSocialDeter(
    list_data=data['train'],
    tokenizer=tokenizer,
    label2id={},
    id2label={},
    max_length=512
)

#
aaa = trial[0]
