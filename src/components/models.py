import os

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, CanineModel

from .loss import NegativeSamplingBCELoss
from .common import MODELS, ModelTypeT

class CanineForMultiLabelClassificationConfig(PretrainedConfig):
    model_type = "CanineMultiLabelClassifier"
    classes: list[str] = ()
    negative_sampling = False

    def __init__(self, classes=(), negative_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
        self.negative_sampling = negative_sampling


class CanineForMultiLabelClassification(PreTrainedModel):
    config_class = CanineForMultiLabelClassificationConfig

    def __init__(self, config: CanineForMultiLabelClassificationConfig):
        super().__init__(config)
        self.config = config
        self.canine = CanineModel.from_pretrained("google/canine-c")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.canine.config.hidden_size, len(self.config.classes))
        self.loss = NegativeSamplingBCELoss(
            config.classes) if config.negative_sampling else nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Quick fix for bug in transformers package
        kwargs.pop("num_items_in_batch", None)
        outputs = self.canine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())

            return {"loss": loss, "logits": logits}

        return {"logits": logits}


class LangIDMultiLabelClassificationConfig(PretrainedConfig):
    model_type = "LangIDMultiLabelClassifier"
    classes: list[str] = ()
    negative_sampling = False
    model = "byt5"

    def __init__(self, model: ModelTypeT = "byt5", classes=(), negative_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.classes = classes
        self.negative_sampling = negative_sampling


class LangIDMultiLabelClassification(PreTrainedModel):
    config_class = LangIDMultiLabelClassificationConfig

    def __init__(self, config: LangIDMultiLabelClassificationConfig):
        super().__init__(config)
        self.config = config
        self.model = MODELS[config.model].model_class.from_pretrained(MODELS[config.model].type, token=os.environ.get("HUGGINGFACE_TOKEN"))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.model.config.hidden_size, len(self.config.classes))
        self.loss = NegativeSamplingBCELoss(
            config.classes) if config.negative_sampling else nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Quick fix for bug in transformers package
        kwargs.pop("num_items_in_batch", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        backbone_output = outputs.last_hidden_state[:, 0, :]
        backbone_output = self.dropout(backbone_output)
        logits = self.classifier(backbone_output)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())

            return {"loss": loss, "logits": logits}

        return {"logits": logits}
