import torch
import torch.nn as nn
from transformers import (AdamW, get_linear_schedule_with_warmup)
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from copy import deepcopy
import pandas as pd

class SBDHTrainer():
    """
    This class packs together following things,
    1. Loss function (defines objective function for our optimization problem)
    2. Optimizer (defines solution methodology to be used for solving formulated optimization problem)
    3. Scheduler (defines method by which we reduce/adapt step size used in our solution method)
    """

    def __init__(
            self,
            device,
            optimizer: str,
            learning_rate: float,
            weight_decay: float,
            model: torch.nn.Module,
            mtl: bool = False,
            adam_epsilon: float = 1e-8,
            gradient_clip: float = 1,
            wandb_project_name: str = 'SBDH_Detection'
    ):

        # STEP 1: define loss function
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(device)

        # STEP 2: define optimizer
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.grouped_parameters = self.get_parameter_groups(
            model=model,
            weight_decay=weight_decay
        )
        # @TODO: need to generalize following line (how to automatically use different optimizers)
        self.optimizer = AdamW(
            self.grouped_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            #epsilon=adam_epsilon,
        )

        """
        I think it'll better to define scheduler in training loop
        
        # STEP 3: define scheduler
        # @TODO: need to generalize following line (how to automatically use different schedulers)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=scheduler_warmup_steps,
            num_training_steps=total_training_steps
        )
        """


        # other important attributes
        self.device = device
        self.mtl = mtl
        self.gradient_clip = gradient_clip
        self.wandb_project_name = wandb_project_name

        return

    def get_parameter_groups(
            self,
            model: torch.nn.Module,
            weight_decay: float
    ) -> list:

        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in self.no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.0}
        ]

        return grouped_parameters

    def measure_performance(
            self,
            labels: list,
            predictions: list,
            save_conf_mat: bool = False,
    ) -> dict:

        # @TODO: don't evaluate for pading

        metrics = {
            'f1-score': 0,
            'precision': 0,
            'recall': 0,
            'accuracy': 0
        }

        for idx, y_hat in enumerate(predictions):

            # get class wise metrics
            class_wise = classification_report(
                y_true=labels[idx],
                y_pred=y_hat,
                output_dict=True
            )

            # get average value of f1
            metrics['f1-score'] += class_wise['weighted avg']['f1-score']
            metrics['precision'] += class_wise['weighted avg']['precision']
            metrics['recall'] += class_wise['weighted avg']['recall']
            metrics['accuracy'] += class_wise['accuracy']


            # save confusion matrix
            if save_conf_mat:
                metrics['confusion_matrix'] = confusion_matrix(
                    y_true=labels,
                    y_pred=predictions
                )

        # take average across batch
        for metric in metrics:
            metrics[metric] /= (idx+1)

        return metrics

    def evaluate(
            self,
            dataloader,
            model: torch.nn.Module,
    ) -> dict:
        model.eval()

        # save results in dictionary
        eval_results = {
            'metrics_sbdh': {
                'f1-score': 0,
                'precision': 0,
                'recall': 0,
                'accuracy': 0
            },
            'metrics_umls': {
                'f1-score': 0,
                'precision': 0,
                'recall': 0,
                'accuracy': 0
            },
        }

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):

                # unroll
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_sbdh = batch['labels_sbdh'].to(self.device)
                target_umls = batch['labels_umls'].to(self.device)

                # forward pass, loss calculation and argmax for predictions
                logits_sbdh, logits_umls = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                if self.mtl:
                    # calculate loss
                    # @TODO: will need to check following line during code review
                    # (I have checked it once, changing shapes to match CrossEntropyLoss definition)
                    # loss_sbdh = self.criterion(logits_sbdh.permute(0, 2, 1), target_sbdh)
                    # loss_umls = self.criterion(logits_umls.permute(0, 2, 1), target_umls)

                    # compute predictions
                    predictions_sbdh = np.argmax(logits_sbdh.detach().cpu().numpy(), axis=-1).tolist()
                    predictions_umls = np.argmax(logits_umls.detach().cpu().numpy(), axis=-1).tolist()

                else:
                    # calculate loss
                    # @TODO: will need to check following line during code review
                    #  (I have checked it once, changing shapes to match CrossEntropyLoss definition)
                    # loss_sbdh = self.criterion(logits_sbdh.permute(0, 2, 1), target_sbdh)
                    # loss_umls = torch.tensor(0.0).to(self.device)

                    # compute predictions
                    predictions_sbdh = np.argmax(logits_sbdh.detach().cpu().numpy(), axis=-1).tolist()
                    predictions_umls = []

                # update evaluation metrics
                metrics_sbdh = self.measure_performance(
                    labels=target_sbdh.detach().numpy().tolist(),
                    predictions=predictions_sbdh
                )
                for metric in eval_results['metrics_sbdh']:
                    eval_results['metrics_sbdh'][metric] += metrics_sbdh[metric]

                if self.mtl:
                    metrics_umls = self.measure_performance(
                        labels=target_umls.detach().numpy().tolist(),
                        predictions=predictions_umls
                    )
                    for metric in eval_results['metrics_umls']:
                        eval_results['metrics_umls'][metric] += metrics_umls[metric]

        # take average across all batches
        for task in eval_results:
            for metric in eval_results[task]:
                eval_results[task][metric] /= (idx + 1)

        return eval_results

    def train(
            self,
            dataloader_train,
            dataloader_val,
            dataloader_test,
            model: torch.nn.Module,
            num_epochs: int,
            total_training_steps: int,
            scheduler_warmup_fraction: float,
            eval_every_steps: int,
    ):

        # retrieve optimizer
        optimizer = self.optimizer

        # define scheduler
        total_training_steps = min(total_training_steps, num_epochs*len(dataloader_train))
        scheduler_warmup = int(scheduler_warmup_fraction*total_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=scheduler_warmup,
            num_training_steps=total_training_steps
        )

        # progress bar
        progress_bar = tqdm(range(total_training_steps))
        global_step = 0
        best_model = None
        best_val_f1 = 0
        best_val_results = None

        # iterate over batches
        for epoch in range(num_epochs):
            model.train()
            for batch in dataloader_train:
                global_step += 1

                # clear gradients
                model.zero_grad()

                # unroll
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_sbdh = batch['labels_sbdh'].to(self.device)
                target_umls = batch['labels_umls'].to(self.device)

                # forward pass, loss calculation and argmax for predictions
                logits_sbdh, logits_umls = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                if self.mtl:
                    # calculate loss
                    # @TODO: will need to check following line during code review
                    #  (I have checked it once, changing shapes to match CrossEntropyLoss definition)
                    loss_sbdh = self.criterion(logits_sbdh.permute(0, 2, 1), target_sbdh)
                    loss_umls = self.criterion(logits_umls.permute(0, 2, 1), target_umls)

                    # compute predictions
                    predictions_sbdh = np.argmax(logits_sbdh.detach().cpu().numpy(), axis=-1).tolist()
                    predictions_umls = np.argmax(logits_umls.detach().cpu().numpy(), axis=-1).tolist()

                else:
                    # calculate loss
                    # @TODO: will need to check following line during code review
                    #  (I have checked it once, changing shapes to match CrossEntropyLoss definition)
                    loss_sbdh = self.criterion(logits_sbdh.permute(0, 2, 1), target_sbdh)
                    loss_umls = torch.tensor(0.0).to(self.device)

                    # compute predictions
                    predictions_sbdh = np.argmax(logits_sbdh.detach().cpu().numpy(), axis=-1).tolist()
                    predictions_umls = []

                # back propagation
                loss_sbdh.backward(retain_graph=True)
                if self.mtl:
                    loss_umls.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

                # update parameters and update lr
                optimizer.step()
                scheduler.step()
                progress_bar.update(1)

                # save info for wandb
                wandb.log(
                    {
                        'NER_training_loss': loss_sbdh,
                        'NEN_training_loss': loss_umls if loss_umls is not None else 0,
                        'Learning_rate': optimizer.param_groups[0]['lr'],
                        'Epoch': epoch,
                    },
                    step=global_step,
                )

                # evaluation
                if global_step%eval_every_steps == 0:

                    # evaluate
                    eval_output = self.evaluate(
                        dataloader=dataloader_val,
                        model=model
                    )


                    # save info to wandb
                    wandb.log(
                        {
                            'NER_f1-score': eval_output['metrics_sbdh']['f1-score'],
                            'NER_precision': eval_output['metrics_sbdh']['precision'],
                            'NER_recall': eval_output['metrics_sbdh']['recall'],
                            'NER_accuracy': eval_output['metrics_sbdh']['accuracy'],

                            'NEN_f1-score': eval_output['metrics_umls']['f1-score'],
                            'NEN_precision': eval_output['metrics_umls']['precision'],
                            'NEN_recall': eval_output['metrics_umls']['recall'],
                            'NEN_accuracy': eval_output['metrics_umls']['accuracy'],
                        },
                        step=global_step,
                    )

                    # update best model
                    if best_val_f1 < eval_output['metrics_sbdh']['f1-score']:
                        best_model = model
                        best_val_results = eval_output

                # break if max training steps are reached
                if global_step >= total_training_steps:
                    break

            # break if max training steps are reached
            if global_step >= total_training_steps:
                break

        # test model
        if best_model is None:
            best_model = model

        test_output = self.evaluate(
            dataloader=dataloader_test,
            model=best_model
        )

        return best_model, best_val_results, test_output