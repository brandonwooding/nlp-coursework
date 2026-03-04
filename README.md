# NLP Coursework — PCL Detection (COMP70016)

Detecting Patronizing and Condescending Language (PCL) using transformer-based ensemble models.

## Repository Structure

```
nlp-coursework/
├── BestModel/                              # Best model: ensemble inference
│   ├── NLP_Ensemble_Model_Inference.ipynb  # ← BEST MODEL (ensemble inference pipeline)
│   ├── ensemble_config.json                # Ensemble weights and threshold config
│   ├── dev.csv                             # Dev split input data
│   ├── test.csv                            # Test split input data
│   └── results/                            # Model output predictions
│       ├── dev.txt                         # Dev set predictions
│       ├── test.txt                        # Test set predictions
│       └── local_analysis.ipynb            # Results analysis notebook
│
├── COMP70016_NLP_bw725.ipynb              # ← TRAINING PIPELINE (all model experiments)
│
├── data/
│   ├── dontpatronizeme_pcl.tsv            # Main PCL dataset
│   ├── task4_test.tsv                     # Task 4 test set
│   ├── dont_patronize_me.py               # Dataset loader
│   └── load_data.py                       # Data loading utilities
│
├── figures/                               # EDA visualisations
│   ├── eda_technique1_class_distribution.png
│   └── eda_technique2_lexical_comparison.png
│
├── models.py                              # Custom model architectures
└── requirements.txt                       # Python dependencies
```

## Key Files

| File | Description |
|------|-------------|
| [BestModel/NLP_Ensemble_Model_Inference.ipynb](BestModel/NLP_Ensemble_Model_Inference.ipynb) | **Best model** — runs ensemble inference over 6 fine-tuned transformer models (RoBERTa, DeBERTa, multi-task, ordinal) |
| [COMP70016_NLP_bw725.ipynb](COMP70016_NLP_bw725.ipynb) | **Training pipeline** — EDA, model training, evaluation, and ablations |
| [BestModel/results/](BestModel/results/) | **Model outputs** — final dev and test predictions |
