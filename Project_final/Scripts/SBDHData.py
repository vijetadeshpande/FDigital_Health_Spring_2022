import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.sparse
from pdb import set_trace as breakpoint
from collections import Counter
from random import shuffle
import json
from transformers import AutoTokenizer

class MIMICDataLoader():

    def __init__(
            self,
            filepath_data: str = 'preprocessed_data_version_2.json',
            filepath_label2id_sbdh: str = 'label2id_sbdh.json',
            filepath_label2id_umls: str = 'label2id_umls.json',
            tokenizer_name: str = 'bert-base-uncased',
            max_sequence_length: int = 512,
            batch_size: int = 16,
            debug: bool = False
    ):
        """This class saves dataloader of each split as attributes"""

        # STEP 1: read data (only one file having three splits)
        with open(filepath_data, 'r') as f:
            data = json.load(f)

        # STEP 2: in addition to data, we need label maps i.e. a dict to convert
        # label text to an integer and an integer back to it's class text
        # NOTE: we have two kinds of labels, so, label2id will have 2 files/attributes
        # and id2label will have two files/attributes
        with open(filepath_label2id_sbdh, 'r') as f:
            self.label2id_sbdh = json.load(f)
        with open(filepath_label2id_umls, 'r') as f:
            self.label2id_umls = json.load(f)

        # the reverse map i.e. id2label is useful in evaluation/interact
        # when we are required to convert prediction (an integer) to class name
        self.id2label_sbdh = self.get_id2label(self.label2id_sbdh)
        self.id2label_umls = self.get_id2label(self.label2id_umls)

        # save number of classes for both labels
        self.num_classes_sbdh = self.get_num_classes(self.label2id_sbdh)
        self.num_classes_umls = self.get_num_classes(self.label2id_umls)

        # STEP 3: define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # STEP 4: tokenize dataset
        self.dataset_train = MIMICDataset(
            list_data=data['train'] if not debug else data['train'][0:16],
            tokenizer=self.tokenizer,
            label2id_sbdh=self.label2id_sbdh,
            label2id_umls=self.label2id_umls,
            max_length=max_sequence_length
        )
        self.dataset_test = MIMICDataset(
            list_data=data['test'] if not debug else data['test'][0:16],
            tokenizer=self.tokenizer,
            label2id_sbdh=self.label2id_sbdh,
            label2id_umls=self.label2id_umls,
            max_length=max_sequence_length
        )
        self.dataset_validation = MIMICDataset(
            list_data=data['validation'] if not debug else data['validation'][0:16],
            tokenizer=self.tokenizer,
            label2id_sbdh=self.label2id_sbdh,
            label2id_umls=self.label2id_umls,
            max_length=max_sequence_length
        )

        # STEP 5
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        self.dataloader_validation = DataLoader(
            self.dataset_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        return

    def get_id2label(
            self,
            label2id: dict
    ) -> dict:
        id2label = {}
        for key_, val_ in label2id.items():
            id2label[val_] = key_

        return id2label

    def get_num_classes(
            self,
            id2label: dict
    ) -> int:

        return len(id2label.keys())


class MIMICDataset(Dataset):
    """
    This class takes list of data instances in a specific split (say train) and
    returns tokenized form of the data when __getitemm__ is called.
    """

    def __init__(
            self,
            list_data: list,
            tokenizer,
            label2id_sbdh: dict,
            label2id_umls: dict,
            max_length: int = 512
    ):

        # attributes
        self.data = list_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id_sbdh = label2id_sbdh
        self.label2id_umls = label2id_umls

        return

    def __len__(self):
        """Returns total length or total data instances in the input data"""
        return self.data.__len__()

    def __getitem__(self, item) -> dict:
        """This is the main function of the class. Returns tokenized form of the called item"""

        # get the 'item' index data instance
        example = self.data[item]

        # tokenize the example
        tokenized = self.tokenizer.encode_plus(
            example['words'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
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
                        label_umls = label_umls + '_b' if not label_umls == 'None' else 'None'
                    elif word_id == prev_id:
                        label_sbdh = label_sbdh + '_i'
                        label_umls = label_umls + '_i' if not label_umls == 'None' else 'None'
                    else:
                        print('Labeling logic is not correct')

                #
                labels_sbdh.append(self.label2id_sbdh[label_sbdh])
                labels_umls.append(self.label2id_umls[label_umls])

            # update prev_id
            prev_id = word_id

        # check
        assert len(tokenized['input_ids']) == len(labels_sbdh) == len(labels_umls)

        # change attention mask
        for idx, id_ in enumerate(tokenized['attention_mask']):
            if id_ == self.tokenizer.pad_token_id:
                break
        tokenized['attention_mask'][idx:] = [0] * (self.max_length - idx)

        #
        tokenized['labels_sbdh'] = labels_sbdh
        tokenized['labels_umls'] = labels_umls

        #
        for key_ in tokenized:
            tokenized[key_] = torch.LongTensor(tokenized[key_])

        return tokenized


# test the code
"""
filepath = r'/Users/vijetadeshpande/Downloads/UMass Lowell - Courses/Spring 2022/Foundations in Digital Health/FDigital_Health_Spring_2022/Project_final/Scripts/preprocessed_data_version_2.json'
with open(filepath, 'r') as f:
    data = json.load(f)

with open('label2id_sbdh.json', 'r') as f:
    label2id_sbdh = json.load(f)
with open('label2id_umls.json', 'r') as f:
    label2id_umls = json.load(f)

#
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

trial = MIMICSocialDeter(
    list_data=data['train'],
    tokenizer=tokenizer,
    label2id_sbdh=label2id_sbdh,
    label2id_umls=label2id_umls,
    max_length=512
)

#
aaa = trial[0]
"""



