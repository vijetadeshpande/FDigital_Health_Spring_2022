import torch
import torch.nn as nn
from transformers import (
    WEIGHTS_NAME,
    AutoConfig, AutoModel,
    BertConfig, BertForTokenClassification, BertTokenizer,
    XLMConfig, XLMForTokenClassification, XLMTokenizer,
    DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer,
    RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
)
from collections import OrderedDict

class CustomClassifier(nn.Module):
    """
    This class defines the additional classifier required for the task.
    Inherits from torch.nn.Module
    This can be used in the multi-task learning (NER + NEN) task
    """
    def __init__(
            self,
            num_labels_ner: int = 15,
            num_labels_nen: int = 100,
            in_features: int = 768,
            mtl: bool = False,
    ):
        super(CustomClassifier, self).__init__()

        #
        self.mtl = mtl

        # classifier-1
        self.classifier_ner = nn.Linear(
            in_features=in_features,
            out_features=num_labels_ner
        )

        # classifier-2
        if mtl:
            self.classifier_nen = nn.Linear(
                in_features=in_features,
                out_features=num_labels_ner
            )

        return

    def forward(
            self,
            encoder_output: torch.tensor
    ):

        # task-1
        logits_ner = self.classifier_ner(encoder_output)

        # task-2
        logits_nen = None
        if self.mtl:
            logits_nen = self.classifier_nen

        return logits_ner, logits_nen

class SBDHModel(nn.Module):

    """
    This class defines a model that inherits from torch.nn.Module

    """

    def __init__(
            self,
            pretrained_model: str = 'bert-base-uncased',
            mtl: bool = False,
            num_classes_sbdh: int = 15,
            num_classes_umls: int = 100,
            classifier_num_layers: int = 1,
            classifier_hidden_size: int = 512,
    ):
        super(SBDHModel, self).__init__()

        #
        self.mtl = mtl

        # STEP 1: define model config
        # @TODO: Not sure how much this line will be helpful in MLT, keep it for now
        #self.config = AutoConfig.from_pretrained(
        #    pretrained_model_name_or_path=pretrained_model,
        #    num_labels=num_classes_sbdh
        #)

        # STEP 2: define the pretrained model you want to use
        # @TODO: later, we will need to think how to generalize this line
        self.encoder = AutoModel.from_pretrained(
            pretrained_model,
        )

        # STEP 3: define custom classifier if we want to
        self.classifier = CustomClassifier(
            num_labels_ner=num_classes_sbdh,
            num_labels_nen=num_classes_umls
        )

        return

    def forward(
            self,
            input_ids: torch.tensor,
            attention_mask: torch.tensor,
            token_type_ids: torch.tensor = None
    ):

        # encoder forward pass
        encodings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # classifier forward pass, if defined
        logits_sbdh, logits_umls = self.classifier(
            encoder_output=encodings.last_hidden_state.to(input_ids.device)
        )

        return logits_sbdh, logits_umls

class n2c2Model(nn.Module):

    """
    This class defines a model that inherits from torch.nn.Module

    """

    def __init__(
            self,
            pretrained_model: str = 'bert-base-uncased',
            num_classes: int = 29,
            classifier_num_layers: int = 1,
            classifier_in_features: int = 768,
            classification_threshold: float = 0.5,
    ):
        super(n2c2Model, self).__init__()

        # STEP 1: define model config
        # @TODO: Not sure how much this line will be helpful in MLT, keep it for now
        #self.config = AutoConfig.from_pretrained(
        #    pretrained_model_name_or_path=pretrained_model,
        #    num_labels=num_classes_sbdh
        #)

        # STEP 2: define the pretrained model you want to use
        # @TODO: later, we will need to think how to generalize this line
        self.encoder = AutoModel.from_pretrained(
            pretrained_model,
        )

        # STEP 3: define custom classifier if we want to
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    (
                        'linear', nn.Linear(
                            in_features=classifier_in_features,
                            out_features=num_classes
                        )
                    ),
                    (
                        'activation', nn.Sigmoid()
                    ),
                ]
            )
        )

        # save threshold for classification
        self.classification_threshold = classification_threshold

        return

    def forward(
            self,
            input_ids: torch.tensor,
            attention_mask: torch.tensor,
            token_type_ids: torch.tensor = None
    ):

        # encoder forward pass
        encodings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # classifier forward pass, if defined
        probabilities = self.classifier(encodings.last_hidden_state.to(input_ids.device))

        # predictions based on the classification threshold selected
        predictions = (probabilities > self.classification_threshold).long().to(input_ids.device)

        return probabilities, predictions
