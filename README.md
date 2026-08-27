# AG News Text Classification

Deep learning solution for classifying news articles (title + description) into
four categories: World, Sports, Business, Sci/Tech, using the AG News
dataset. Two architectures are built, trained, and compared: a static-embedding
BiLSTM and a fine-tuned DistilBERT transformer.

## Problem Statement

News platforms publish thousands of articles a day across many topics. Manually
sorting this volume into categories is slow, inconsistent, and unscalable.
Automatic classification enables real-time content routing, personalized
recommendation, and content moderation at scale.

Classical ML (e.g. TF-IDF + Logistic Regression/Naive Bayes) treats words as
independent sparse features and struggles with:
- Semantic similarity: e.g. "shares", "stocks", "markets" all point to the
  same topic even without exact word overlap with training data.
- Word order and context: e.g. "Apple reports record iPhone sales" vs.
  "apple orchard yield record harvest", where the same word means different
  things depending on context.

Deep learning addresses this via word embeddings and sequence/context modeling.

## Models

| | Model A | Model B |
|---|---|---|
| Architecture | GloVe embeddings (frozen, 100d) + BiLSTM | Fine-tuned `distilbert-base-uncased` |
| Embedding type | Static (non-contextual) | Contextual (transformer, transfer learning) |
| Framework | TensorFlow / Keras | PyTorch / Hugging Face Transformers |
| Head | BiLSTM(64) -> Dropout -> Dense(64, relu) -> Dropout -> Dense(4, softmax) | Pretrained encoder + classification head |
| Training | Adam, early stopping on val loss | AdamW (lr=2e-5), manual training loop, early stopping on val loss |

## Notebook Structure

The full pipeline lives in `AG_News_Classifier.ipynb`:

1. Setup & Data Loading: mount Drive, install deps, load AG News CSVs, map labels
2. Exploratory Data Analysis: class balance, missing/duplicate checks, raw text
   quality, text length distributions, word frequency (Zipf) plot, word clouds,
   top bigrams per class, TF-IDF class similarity heatmap
3. Data Cleaning & Preprocessing: text cleaning; separate pipelines for
   Model A (Keras `Tokenizer` + padding + GloVe embedding matrix) and Model B
   (DistilBERT tokenizer)
4. Model A: GloVe + BiLSTM
5. Model B: Fine-tuned DistilBERT
6. Model Evaluation & Comparison: training curves, test predictions,
   confusion matrices, per-class F1, t-SNE projection (2D/3D) of learned
   representations, error overlap (Venn diagram), qualitative misclassification
   inspection, efficiency/trade-off comparison (params, train/inference time vs. accuracy)
7. Discussion & Conclusion: performance, strengths/weaknesses, error
   behaviour, accuracy-vs-cost trade-offs, and future improvements

## Results / Metrics

Test set results (7,600 held-out AG News articles, 1,900 per class):

| Metric | Model A (GloVe + BiLSTM) | Model B (DistilBERT) |
|---|---|---|
| Test accuracy | 0.918 | 0.941 |
| Macro F1 | 0.92 | 0.94 |
| Parameters | 1.28M | 66.96M |

Per-class F1 (test set):

| Class | Model A precision / recall / F1 | Model B precision / recall / F1 |
|---|---|---|
| World | 0.95 / 0.90 / 0.92 | 0.94 / 0.95 / 0.95 |
| Sports | 0.96 / 0.98 / 0.97 | 0.98 / 0.99 / 0.99 |
| Business | 0.89 / 0.88 / 0.88 | 0.93 / 0.90 / 0.92 |
| Sci/Tech | 0.88 / 0.91 / 0.89 | 0.91 / 0.93 / 0.92 |

Model B (DistilBERT) outperforms Model A on every class, most notably on
Business and Sci/Tech, the two categories that are hardest to separate since
they share overlapping vocabulary (e.g. tech companies' financial news). The
gain comes at a large cost: roughly 52x more parameters, 8x longer training
time, and 2.5x longer inference time than Model A. See section 6 of the
notebook for training curves, confusion matrices, t-SNE projections, and
qualitative error analysis, and section 7 for the full discussion of this
accuracy-versus-cost trade-off.

## Dataset

[AG News](http://groups.di.unipr.it/~fabbri/AGNEWS/): expects `train.csv` and
`test.csv` (columns: `label`, `title`, `description`) at
`/content/drive/MyDrive/ag_news/` when run on Google Colab.

The `train.csv` and `test.csv` files used for this assignment are available
in this Google Drive folder:
[ag_news](https://drive.google.com/drive/folders/1NPHqPM5jxR85RLus4vZFNlqdfn5B7Awk?usp=drive_link).
Copy or shortcut the folder into `MyDrive/ag_news/` before running the
notebook.

## Requirements

Designed to run on Google Colab (GPU runtime recommended). Key dependencies,
installed in-notebook:

```
tensorflow
torch
transformers
datasets
scikit-learn
pandas / numpy
matplotlib / seaborn / plotly
wordcloud
umap-learn
matplotlib-venn
nltk
```

## How to Run

1. Open `SG_News_Classifier.ipynb` in Google Colab.
2. Place `train.csv` and `test.csv` in `MyDrive/ag_news/`.
3. Run all cells top to bottom (Runtime -> Run all). A GPU runtime is
   recommended for reasonable training/inference times, especially for Model B.

## How to Reproduce These Results

1. Use a Colab GPU runtime (Runtime -> Change runtime type -> GPU). Running on
   CPU will still produce the same accuracy/F1 numbers, but training and
   inference times will differ from those reported above.
2. Mount Google Drive and place the original, unmodified AG News `train.csv`
   (120,000 rows) and `test.csv` (7,600 rows) at `MyDrive/ag_news/`, matching
   the path set in section 1 of the notebook.
3. Run all cells in order without skipping any (Runtime -> Run all). Later
   cells depend on variables and preprocessed data created earlier
   (`train_split`, `val_split`, `tokenizer_a`, `embedding_matrix`,
   `bert_tokenizer`, etc.).
4. Do not change `SEED = 42` in section 1. It is applied to `random`, `numpy`,
   `tensorflow`, and `torch` and controls the train/validation split, weight
   initialization, and batch shuffling.
5. Keep the pinned settings from the notebook: GloVe `6B.100d` vectors
   (downloaded in section 3.1), `distilbert-base-uncased` as the Model B
   checkpoint, `BERT_MAX_LEN = 64`, and the early-stopping callbacks
   (`monitor='val_loss', patience=2`) on both models. Changing any of these
   will shift the reported accuracy, F1, and timing figures.
6. The efficiency comparison in section 6.8 times inference on the first 500
   test samples on whatever device the runtime provides (GPU vs. CPU), so
   timing figures (not accuracy/F1) will vary with the hardware used.

Minor run-to-run variation (a fraction of a percentage point in accuracy/F1)
is expected even with a fixed seed, due to non-determinism in GPU kernels and
in DistilBERT's dropout/AdamW updates.

## Limitations

- Trained and evaluated only on AG News, a balanced, four-class dataset of
  short titles and descriptions from 2000s-era news wire text. Performance may
  not transfer to full articles, other domains, more classes, or an
  imbalanced real-world label distribution.
- Model A uses frozen GloVe embeddings (`trainable=False`), so it cannot learn
  task-specific word representations or handle out-of-vocabulary/rare terms
  beyond the fixed pretrained vocabulary.
- Both models truncate/pad inputs to a fixed length (95th-percentile length
  for Model A, 64 tokens for Model B), so longer articles are cut off and may
  lose information relevant to classification.
- Business and Sci/Tech remain the most confused class pair for both models
  (see the confusion matrices and error overlap analysis in section 6),
  reflecting genuine topic overlap (e.g. tech-company earnings reports) that
  neither architecture fully resolves.
- No hyperparameter search was performed; architecture choices (LSTM units,
  dropout rates, learning rates, embedding dimension) are fixed single
  configurations rather than tuned or cross-validated.
- Results come from a single train/validation split and a single training run
  per model, not repeated runs or k-fold cross-validation, so reported metrics
  carry some unquantified variance.
- Model B's higher accuracy comes with substantially higher compute cost
  (~52x parameters, ~8x training time, ~2.5x inference time versus Model A),
  which may make it impractical for latency- or resource-constrained
  deployments.
- The notebook is designed around Google Colab and Google Drive paths; running
  it elsewhere requires adapting the data-loading and dependency-install cells.
