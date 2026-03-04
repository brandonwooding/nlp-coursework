# inference.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from models import MultiTaskRoberta, OrdinalRoberta
import json

# Load config
with open('ensemble_config.json') as f:
    config = json.load(f)
weights = config['weights']
threshold = config['threshold']

# Load tokenizers
roberta_tokenizer = AutoTokenizer.from_pretrained('brandonwooding/patronize-baseline-roberta')
deberta_tokenizer = AutoTokenizer.from_pretrained('brandonwooding/patronize-baseline-deberta')

# Load HuggingFace models
baseline_model = RobertaForSequenceClassification.from_pretrained('brandonwooding/patronize-baseline-roberta')
weighted_model = RobertaForSequenceClassification.from_pretrained('brandonwooding/patronize-weighted-roberta')
soft_model = RobertaForSequenceClassification.from_pretrained('brandonwooding/patronize-soft-roberta')
deberta_model = AutoModelForSequenceClassification.from_pretrained('brandonwooding/patronize-baseline-deberta')

# Load custom models from HuggingFace
multi_path = hf_hub_download(repo_id='brandonwooding/patronize-multitask-roberta', filename='multi_task_roberta.pt')
ordinal_path = hf_hub_download(repo_id='brandonwooding/patronize-ordinal-roberta', filename='ordinal_roberta.pt')

multi_model = MultiTaskRoberta()
multi_model.load_state_dict(torch.load(multi_path, map_location='cpu'))
multi_model.eval()

ordinal_model = OrdinalRoberta()
ordinal_model.load_state_dict(torch.load(ordinal_path, map_location='cpu'))
ordinal_model.eval()

# Set all models to eval mode
baseline_model.eval()
weighted_model.eval()
soft_model.eval()
deberta_model.eval()


def predict_ensemble(df):
    """Takes a dataframe with 'keyword' and 'paragraph' columns, returns predictions and probabilities."""

    # Prepare input text
    texts = (df['keyword'].fillna('').astype(str) + ' : ' + df['paragraph'].fillna('').astype(str)).tolist()

    # Tokenize
    rob_enc = roberta_tokenizer(texts, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    deb_enc = deberta_tokenizer(texts, truncation=True, padding='max_length', max_length=256, return_tensors='pt')

    with torch.no_grad():
        # Standard models
        prob_baseline = softmax(baseline_model(**rob_enc).logits.numpy(), axis=1)[:, 1]
        prob_weighted = softmax(weighted_model(**rob_enc).logits.numpy(), axis=1)[:, 1]
        prob_soft = softmax(soft_model(**rob_enc).logits.numpy(), axis=1)[:, 1]
        prob_deberta = softmax(deberta_model(**deb_enc).logits.numpy(), axis=1)[:, 1]

        # Custom models
        multi_out = multi_model(input_ids=rob_enc['input_ids'], attention_mask=rob_enc['attention_mask'])
        prob_multi = softmax(multi_out['logits'].numpy(), axis=1)[:, 1]

        ordinal_out = ordinal_model(input_ids=rob_enc['input_ids'], attention_mask=rob_enc['attention_mask'])
        prob_ordinal = torch.sigmoid(ordinal_out['logits'][:, 1]).numpy()

    # Weighted average
    weighted_prob = (
        weights['baseline'] * prob_baseline +
        weights['weighted'] * prob_weighted +
        weights['soft'] * prob_soft +
        weights['multi'] * prob_multi +
        weights['ordinal'] * prob_ordinal +
        weights['deberta'] * prob_deberta
    )

    preds = (weighted_prob > threshold).astype(int)
    return preds, weighted_prob


def save_predictions(preds, filename):
    """Save predictions to txt file, one per line."""
    with open(filename, 'w') as f:
        for pred in preds:
            f.write(f"{pred}\n")


if __name__ == '__main__':
    # Load your data
    dev_df = pd.read_csv('../data/dev.csv')
    test_df = pd.read_csv('../data/test.csv')

    # Get predictions
    dev_preds, dev_probs = predict_ensemble(dev_df)
    test_preds, test_probs = predict_ensemble(test_df)

    # Save to txt
    save_predictions(dev_preds, 'dev.txt')
    save_predictions(test_preds, 'test.txt')

    # If dev set has labels, print F1
    if 'label' in dev_df.columns:
        from sklearn.metrics import f1_score
        print(f"Dev F1: {f1_score(dev_df['label'].values, dev_preds):.4f}")

    print(f"Dev predictions saved: {len(dev_preds)}")
    print(f"Test predictions saved: {len(test_preds)}")