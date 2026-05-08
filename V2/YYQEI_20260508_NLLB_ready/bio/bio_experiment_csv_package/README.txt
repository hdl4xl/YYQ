# BIO experiment CSV package

Files:
- bio_spans_experiment.csv: one row per sentence; keeps spans_json and labels_json.
- bio_sequence_experiment.csv: one row per sentence; tokens_json and labels_json are aligned character-level arrays.
- bio_char_long_experiment.csv: one row per character; columns are id, split, char_index, char, bio_label.
- *_train/dev/test.csv: split-specific versions.
- bio_experiment_validation_summary.csv: quality checks.
- bio_entity_type_counts.csv and bio_char_label_counts.csv: label distributions.

Recommended usage:
- For HuggingFace-style sequence labeling: use bio_sequence_train/dev/test.csv.
- For CRF/BiLSTM/character-level NER loaders: use bio_char_long_train/dev/test.csv and group by id.
- For span-based evaluation or data inspection: use bio_spans_train/dev/test.csv.

The label scheme is BIO at Chinese character level. `start` is inclusive and `end` is exclusive in spans_json.
