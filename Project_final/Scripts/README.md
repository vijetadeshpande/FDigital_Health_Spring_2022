# NER for Social and Behavioral Determinants of Health from Clinicial Notes

## Experiments tracked
`test/ner/macro avg` is the macro avg F1-score.

[Weights and biases page](https://wandb.ai/mohdelgaar/hiba-sbdh)

[Run with F1-score of 96.52](https://wandb.ai/mohdelgaar/hiba-sbdh/runs/3eiv92ok)

## How to run
`python SBDHExperiments.py --pretrained_model dmis-lab/biobert-v1.1 --dataset_name hiba --wandb_project hiba-sbdh --batch_size 32 --learning_rate 5e-5 --num_train_epochs 10`

## Relevant files
- `SBDHData.py`
- `SBDHModel.py`
- `SBDHTrainer.py`
- `SBDHExperiments.py`
- `DataPreprocessing_NER.py`
