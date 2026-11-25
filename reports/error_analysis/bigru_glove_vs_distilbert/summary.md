# Error analysis: bigru_glove_vs_distilbert

- Test resumes: 249; wrong predictions: 32 (12.9%)
- Accuracy 0.8715, macro-F1 0.8273
- For 31% of the errors the true category was still in the model's top 3

## Weakest categories

| category   |   precision |   recall |    f1 |   support |
|:-----------|------------:|---------:|------:|----------:|
| AUTOMOBILE |       0     |    0     | 0     |         3 |
| BPO        |       0.5   |    0.5   | 0.5   |         2 |
| ADVOCATE   |       0.615 |    0.667 | 0.64  |        12 |
| ARTS       |       0.667 |    0.727 | 0.696 |        11 |
| FITNESS    |       0.8   |    0.667 | 0.727 |        12 |
| APPAREL    |       0.7   |    0.778 | 0.737 |         9 |

## Strongest categories

| category   |   precision |   recall |   f1 |   support |
|:-----------|------------:|---------:|-----:|----------:|
| TEACHER    |           1 |        1 |    1 |        10 |
| HR         |           1 |        1 |    1 |        11 |
| FINANCE    |           1 |        1 |    1 |        12 |
| DESIGNER   |           1 |        1 |    1 |        10 |
| ACCOUNTANT |           1 |        1 |    1 |        12 |

## Most frequent confusions

| true        | pred                 |   count |   share of that true class |
|:------------|:---------------------|--------:|---------------------------:|
| FITNESS     | ADVOCATE             |       3 |                       0.25 |
| ADVOCATE    | ARTS                 |       2 |                       0.17 |
| ADVOCATE    | CONSULTANT           |       1 |                       0.08 |
| AGRICULTURE | ADVOCATE             |       1 |                       0.17 |
| ADVOCATE    | FITNESS              |       1 |                       0.08 |
| APPAREL     | AVIATION             |       1 |                       0.11 |
| APPAREL     | BANKING              |       1 |                       0.11 |
| ARTS        | AUTOMOBILE           |       1 |                       0.09 |
| ARTS        | AVIATION             |       1 |                       0.09 |
| ARTS        | HEALTHCARE           |       1 |                       0.09 |
| AUTOMOBILE  | BPO                  |       1 |                       0.33 |
| AUTOMOBILE  | BUSINESS-DEVELOPMENT |       1 |                       0.33 |

## Automatic error tags (one error can have several)

| auto_tags                   |   errors |
|:----------------------------|---------:|
| confident but wrong         |       29 |
| true label not in top-3     |       22 |
| true label in top-3         |       10 |
| title names predicted class |        4 |
| title names true class      |        3 |
| low confidence              |        3 |

## Manual step

Open `errors_to_label.csv`, read 30-50 errors and fill in `error_type` using a fixed list, e.g.:
overlapping categories / mixed-career resume / label looks wrong / too short or noisy / generic wording / other.
Then count the types and describe 3 examples in the report.

## Comparison of the two models

| case                   |   test resumes |
|:-----------------------|---------------:|
| both wrong             |             23 |
| only bigru_glove wrong |              9 |
| only distilbert wrong  |             21 |
| both correct           |            196 |

Errors both models make are the hard cases (see `errors_shared_by_both.csv`).

## Error types (first pass, from reading the start of each resume; please verify)

| error_type (first pass, check)            |   errors | share   |
|:------------------------------------------|---------:|:--------|
| label does not match resume content       |       15 | 47%     |
| overlapping / adjacent fields             |        7 | 22%     |
| generic resume (admin / customer service) |        6 | 19%     |
| mixed career                              |        2 | 6%      |
| genuine model mistake                     |        2 | 6%      |

Examples:

- **Label does not match content:** resume 24219583 is a registered nurse resume labelled FITNESS; resume 17864043 is a corporate recruiter (HR work) labelled HEALTHCARE.
- **Overlapping fields:** resume 45077654 is a claims expert in the healthcare industry, labelled BPO and predicted HEALTHCARE.
- **Generic resume:** resume 23804341 is a receptionist / administrative assistant with no field-specific words, labelled ADVOCATE.
