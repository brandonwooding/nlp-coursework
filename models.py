import torch
import torch.nn as nn
from transformers import RobertaModel

class MultiTaskRoberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        # Two separate heads
        self.binary_head = nn.Linear(768, 2)       # binary classification
        self.severity_head = nn.Linear(768, 1)      # regression 0-4

    def forward(self, input_ids, attention_mask, labels=None, severity_labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)

        binary_logits = self.binary_head(pooled)
        severity_pred = self.severity_head(pooled).squeeze(-1)

        loss = None
        if labels is not None and severity_labels is not None:
            alpha = 0.7
            binary_loss = nn.CrossEntropyLoss()(binary_logits, labels)
            severity_loss = nn.MSELoss()(severity_pred, severity_labels)
            loss = alpha * binary_loss + (1 - alpha) * severity_loss

        return {'loss': loss, 'logits': binary_logits}
    

class OrdinalRoberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.ordinal_head = nn.Linear(768, 4)  # P(>=1), P(>=2), P(>=3), P(>=4)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        logits = self.ordinal_head(pooled)

        loss = None
        if labels is not None:
            # Convert orig_label to ordinal targets
            # e.g. orig_label=2 -> [1, 1, 0, 0]
            targets = torch.zeros_like(logits)
            for i in range(4):
                targets[:, i] = (labels > i).float()
            loss = nn.BCEWithLogitsLoss()(logits, targets)

        return {'loss': loss, 'logits': logits}