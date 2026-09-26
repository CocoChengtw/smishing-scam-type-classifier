# Results

## Leakage check

- Random row split: macro-F1 **0.829**
- Template-grouped split: macro-F1 **0.800**

## Model comparison (template-grouped test set)

| Model | Macro-F1 (95% CI) | Macro-F1, one msg per template | Accuracy |
|---|---|---|---|
| flat_unweighted | 0.795 (0.770–0.817) | 0.771 | 0.832 |
| flat_balanced | 0.800 (0.777–0.818) | 0.775 | 0.828 |
| flat_balanced_template_cap | 0.801 (0.777–0.819) | 0.777 | 0.831 |
| hierarchical_template_cap | 0.797 (0.774–0.815) | 0.769 | 0.829 |

Learned confusion groups: [['banking', 'government', 'others', 'spam', 'telecom', 'wrong_number'], ['delivery'], ['family_impersonation']]

## Per-class (flat_balanced_template_cap)

|                      |   precision |   recall |    f1 |   support |
|:---------------------|------------:|---------:|------:|----------:|
| banking              |       0.925 |    0.915 | 0.92  |      2866 |
| delivery             |       0.937 |    0.904 | 0.92  |       769 |
| family_impersonation |       0.979 |    0.979 | 0.979 |        48 |
| government           |       0.803 |    0.803 | 0.803 |       589 |
| others               |       0.665 |    0.67  | 0.667 |      1337 |
| spam                 |       0.621 |    0.693 | 0.655 |       349 |
| telecom              |       0.808 |    0.785 | 0.796 |       446 |
| wrong_number         |       0.593 |    0.761 | 0.667 |        71 |

## Per-language

| language   |    n |   macro_f1 |
|:-----------|-----:|-----------:|
| English    | 4021 |      0.816 |
| Spanish    | 1043 |      0.788 |
| Dutch      |  419 |      0.649 |
| French     |  221 |      0.78  |
| Italian    |  165 |      0.688 |
| German     |  153 |      0.544 |

## Review routing

|   threshold |   coverage |   accuracy |   macro_f1 |
|------------:|-----------:|-----------:|-----------:|
|        0    |      1     |      0.831 |      0.801 |
|        0.5  |      0.872 |      0.887 |      0.858 |
|        0.6  |      0.787 |      0.919 |      0.887 |
|        0.7  |      0.709 |      0.946 |      0.914 |
|        0.8  |      0.605 |      0.968 |      0.936 |
|        0.9  |      0.465 |      0.982 |      0.949 |
|        0.95 |      0.355 |      0.994 |      0.979 |
