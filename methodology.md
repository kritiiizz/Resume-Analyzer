## Methodology

### 1. Classical Machine Learning Baselines

Four classical machine learning models were implemented as baselines for the 24-class resume classification task: Dummy Classifier, Multinomial Naive Bayes (NB), Logistic Regression (LR), and Linear Support Vector Machine (LinearSVC). These models were selected because they provide established, computationally efficient baselines for text classification and enable a meaningful comparison with Transformer-based approaches.

Two sparse text representations were employed: Bag-of-Words (BoW), which represents documents through token occurrence frequencies, and Term Frequency–Inverse Document Frequency (TF-IDF), which reduces the influence of frequently occurring terms while emphasizing more discriminative features. Both representations use word-level unigrams and bigrams, with a minimum document frequency of 2 and a maximum document frequency of 0.95. Sublinear term-frequency scaling is additionally applied to TF-IDF. A customized tokenization pattern preserves technical skill expressions such as C++, C#, and .NET.

| Model | Representation | Hyperparameter search space |
|---|---|---|
| Dummy Classifier | BoW | Most frequent class |
| Multinomial NB | BoW | α ∈ {0.5, 1, 2} |
| Logistic Regression | TF-IDF | C ∈ {0.1, 1, 10}; class weight ∈ {None, balanced} |
| LinearSVC | TF-IDF | C ∈ {0.1, 1, 10}; class weight ∈ {None, balanced} |

All vectorizers were fitted exclusively on the training set to prevent information leakage. Hyperparameters were selected through grid search using validation Macro-F1 as the selection criterion. The test set was not used for hyperparameter tuning.

### 2. DistilBERT-based Classification

#### Model architecture and input representation

The pretrained `distilbert-base-uncased` model was fine-tuned for 24-class resume classification. DistilBERT was selected because it provides contextual text representations while requiring fewer computational resources than standard BERT. Both the pretrained encoder and the classification head were fine-tuned.

A major challenge in resume classification is the model's maximum input length of 512 tokens, including two special tokens. A WordPiece token-length audit conducted on the original dataset revealed that 1,779 out of 1,984 training resumes (89.67%) and 220 out of 248 validation resumes (88.71%) exceeded the available budget of 510 content tokens.

To address this limitation, the Head+Tail truncation strategy proposed by Sun et al. (2019) was adopted. For documents exceeding the token budget, the first 128 and last 382 content tokens were retained, together with the [CLS] and [SEP] special tokens, resulting in a maximum sequence length of 512 tokens. Shorter documents were preserved in their entirety.

This strategy was chosen to retain information from both the beginning and the end of a resume rather than discarding all content beyond the first 510 tokens. These sections may contain complementary occupational information, professional summaries, educational qualifications, and technical skills.

#### Text preprocessing

Light preprocessing was applied before WordPiece tokenization, including Unicode normalization (NFKC), whitespace normalization, lowercasing, and personally identifiable information (PII) masking for email addresses, telephone numbers, and URLs. Stop-word removal and lemmatization were excluded to preserve natural linguistic structures and contextual information consistent with the model's pretraining conditions.

#### Training configuration

| Hyperparameter | Configuration |
|---|---|
| Pretrained model | DistilBERT-base-uncased |
| Number of output classes | 24 |
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay | 0.01 |
| Learning rate scheduler | Linear decay |
| Warmup ratio | 10% |
| Maximum epochs | 4 |
| Training batch size | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Evaluation batch size | 4 |
| Maximum sequence length | 512 tokens |
| Early stopping patience | 2 epochs |
| Training precision | FP32 |
| Random seeds | 42, 123, 2026 |
| Checkpoint selection | Validation Macro-F1 |
| Computing environment | Apple M3 Pro, PyTorch MPS |

Training and validation metrics were recorded after each epoch, and the checkpoint achieving the highest validation Macro-F1 was retained. Experiments were repeated using three training seeds while maintaining identical dataset partitions to examine sensitivity to training randomness.

### 3. Evaluation Protocol

To ensure consistent and reproducible comparisons, the dataset was cleaned by removing invalid records and exact duplicate resumes, while conflicting labels associated with identical normalized texts were quarantined. The resulting dataset contained 2,481 resumes across 24 occupational categories and was partitioned using a stratified 80/10/10 train–validation–test split with a fixed random seed of 42, yielding 1,984 training, 248 validation, and 249 test samples.

All models shared the same frozen sample identifiers and label mappings. Performance was evaluated using Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1, and Top-3 Accuracy. Per-class classification reports and both raw and row-normalized confusion matrices were also generated to support error analysis.

Macro-F1 was selected as the primary evaluation metric because the dataset exhibits class imbalance, with some occupational categories containing substantially fewer samples than others. Unlike accuracy or support-weighted metrics, Macro-F1 assigns equal importance to each category.

For LinearSVC, class rankings were obtained from decision scores rather than probabilities. DistilBERT predictions were ranked using softmax outputs, which were treated as uncalibrated probabilities. Model selection and early stopping relied exclusively on validation Macro-F1. Test metrics recorded during extended training experiments were used for diagnostic purposes rather than checkpoint selection.

### 4. No-Title Experiment

A controlled experiment was conducted to investigate the extent to which resume classification models rely on explicit job-title information. The original resume text (`Resume_str`) was replaced with `text_no_header`, obtained through the team's HTML-based header removal procedure.

All five classification pipelines were independently retrained using the same model architectures, hyperparameter search spaces, training configurations, and random seeds (42, 123, and 2026). Sample identifiers, category labels, and train–validation–test membership were preserved across both experimental conditions.

This paired experimental design isolates the effect of header removal while controlling for dataset composition and model configuration. It enables the comparison of model performance with and without explicit occupational headers and provides insight into the models' reliance on title-related lexical cues versus the remaining resume content.

### Reference

Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to Fine-Tune BERT for Text Classification? *Chinese Computational Linguistics (CCL 2019)*, 194–206. https://arxiv.org/abs/1905.05583
