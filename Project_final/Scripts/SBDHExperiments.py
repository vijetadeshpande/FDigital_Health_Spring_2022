# Python standard library imports
import argparse
import logging
import os
import torch
import transformers
import random
import wandb

# components of study
from SBDHData import (
    n2c2DataLoader,
    MIMICDataLoader
)
from SBDHModel import (
    SBDHModel,
    n2c2Model,
)
from SBDHTrainer import (
    SBDHTrainer,
    n2c2Trainer,
)

def parse_args():
    """
    Set hyperparameter values here
    """
    parser = argparse.ArgumentParser(description="Detection of social determinants of health in MIMIC")

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default='bert_results',
        help=("Where to store the final model. "
              "Should contain the source and target tokenizers in the following format: "
              r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
              "Both of these should be directories containing tokenizer.json files."
              ),
    )

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="n2c2",
        choices=['hiba', 'n2c2'],
        help="Path of the preprocessed dataset",
    )

    parser.add_argument(
        "--filepath_data",
        type=str,
        default="n2c2_preprocessed.json",
        help="Path of the preprocessed dataset",
    )
    parser.add_argument(
        "--filepath_label2id_sbdh",
        type=str,
        default="n2c2_label2id.json",
        help="filepath for label-to-labelid map for SBDH labels",
    )
    parser.add_argument(
        "--filepath_label2id_umls",
        type=str,
        default="label2id_umls.json",
        help="filepath for label_umls map",
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        action="store_true",
        help="Whether to use a small subset of the dataset for debugging.",
    )

    # Model arguments
    parser.add_argument(
        "--multi_task_learning",
        default=False,
        type=bool,
        help="Indicator for defining the classifier architecture",
    )
    parser.add_argument(
        "--pretrained_model",
        default='dmis-lab/biobert-v1.1',
        type=str,
        help="Name of the pretrained language model for defining encoder",
        choices=[
            'bert-base-uncased',
            'dmis-lab/biobert-v1.1',
            'emilyalsentzer/Bio_ClinicalBERT',
            'emilyalsentzer/Bio_Discharge_Summary_BERT'
        ]
    )
    parser.add_argument(
        "--tokenizer",
        default='dmis-lab/biobert-v1.1',
        type=str,
        help="Tokenizer to correctly format preprocessed data for pretrained model input",
        choices=[
            'bert-base-uncased',
            'dmis-lab/biobert-v1.1',
            'emilyalsentzer/Bio_ClinicalBERT',
            'emilyalsentzer/Bio_Discharge_Summary_BERT'
        ]
    )
    parser.add_argument(
        "--classifier_num_layers",
        default=1,
        type=int,
        help="Number of hidden layers in the classifier",
    )
    parser.add_argument(
        "--classifier_hidden_size",
        default=768,
        type=int,
        help="Hidden size of the classifier",
    )
    parser.add_argument(
        "--binary_classification_threshold",
        default=0.5,
        type=float,
        help="Threshold for binary classification problem",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=300,
        help="The maximum total sequence length for input sentence/s"
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
             "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=None,
        help="Overwrite the cached training and evaluation sets",
    )

    # Training arguments
    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device (cuda or cpu) on which the code should run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='AdamW',
        help="Optimizer to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.0,
        type=float,
        help="Dropout rate of the Transformer encoder",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=4,
        help="Perform evaluation every n network updates.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=transformers.SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--scheduler_warmup_fraction",
        type=float,
        default=0.2,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--wandb_project",
        default="n2c2_NER",
        help="wandb project name to log metrics to"
    )

    args = parser.parse_args()

    return args


def main():

    # parse args
    args = parse_args()

    # check
    assert args.pretrained_model == args.tokenizer

    # init wandb
    wandb.init(
        project=args.wandb_project,
        config=args,
    )

    # make result dir if does not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # prepare data
    if args.dataset_name == 'n2c2':
        dataloders = n2c2DataLoader(
            filepath_data=args.filepath_data,
            filepath_label2id=args.filepath_label2id_sbdh,
            tokenizer_name=args.tokenizer,
            max_sequence_length=args.max_seq_length,
            batch_size=args.batch_size,
            debug=args.debug
        )
    else:
        dataloders = MIMICDataLoader(
            filepath_data=args.filepath_data,
            filepath_label2id_sbdh=args.filepath_label2id_sbdh,
            filepath_label2id_umls=args.filepath_label2id_umls,
            tokenizer_name=args.tokenizer,
            max_sequence_length=args.max_seq_length,
            batch_size=args.batch_size,
            debug=args.debug
        )

    # define model
    if args.dataset_name == 'n2c2':
        model = n2c2Model(
            pretrained_model=args.pretrained_model,
            num_classes=dataloders.num_classes_sbdh,
            classifier_in_features=args.classifier_hidden_size,
            classification_threshold=args.binary_classification_threshold,
        )
    else:
        model = SBDHModel(
            pretrained_model=args.pretrained_model,
            mtl=args.multi_task_learning,
            num_classes_sbdh=dataloders.num_classes_sbdh,
            num_classes_umls=dataloders.num_classes_umls,
            classifier_num_layers=args.classifier_num_layers,
            classifier_hidden_size=args.classifier_hidden_size,
        )
    model.to(args.device)

    # prepare trainer
    if args.dataset_name == 'n2c2':
        trainer = n2c2Trainer(
            device=args.device,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model=model,
        )
    else:
        trainer = SBDHTrainer(
            device=args.device,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model=model,
            mtl=args.multi_task_learning,
        )

    # start training
    best_model, validation_results, test_results = trainer.train(
        dataloader_train=dataloders.dataloader_train,
        dataloader_val=dataloders.dataloader_validation,
        dataloader_test=dataloders.dataloader_test,
        model=model,
        num_epochs=args.num_train_epochs,
        total_training_steps=args.max_train_steps,
        scheduler_warmup_fraction=args.scheduler_warmup_fraction,
        eval_every_steps=args.eval_every_steps,
    )

    return


if __name__ == "__main__":
    # set environment variable for gpu index we want to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
