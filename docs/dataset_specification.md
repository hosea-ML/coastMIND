# CoastMind Dataset Specification

## 1. Purpose

The purpose of the CoastMind dataset is to provide a carefully curated body of coastal-science text for training and evaluating a small decoder-only Transformer language model built with PyTorch.

The dataset will help the project explore how a language model learns scientific vocabulary, relationships between concepts, and patterns in coastal-science explanations.

The dataset is designed for education and experimentation, not for replacing scientific models, field measurements, or expert judgement.

## 2. Project Scope

CoastMind will initially focus on coastal and marine science, with particular attention to coastal processes and environmental change.

The initial content areas are:

- Coastal erosion and accretion
- Sediment transport and sediment dynamics
- Tides, waves, and ocean currents
- Estuaries and tidal inlets
- Shoreline change and coastal geomorphology
- Coastal flooding and inundation
- Remote sensing and Earth observation
- Coastal monitoring and environmental vulnerability

## 3. Data Strategy

CoastMind will use a hybrid data strategy combining appropriately licensed real-world educational or scientific text with carefully designed synthetic examples.

### 3.1 Real-World Text

Potential sources include open educational resources, public-domain scientific material, openly licensed reports, and other sources whose terms permit the intended use.

Each source must be checked for its licence, attribution requirements, permitted reuse, and suitability for machine-learning training.

### 3.2 Synthetic Text

Synthetic examples will be written or generated to explain coastal-science concepts, illustrate relationships between variables, and provide controlled examples for experiments.

Synthetic text must not be presented as an authentic field observation, published scientific finding, or measured result.

### 3.3 Source Separation

Real-world and synthetic records will be identifiable in the dataset. This will allow experiments to compare their contributions and investigate whether synthetic examples introduce misleading patterns.

## 4. Data Record Design

Each text record should have a stable identifier and appropriate provenance metadata.

Proposed fields:

| Field | Description |
|---|---|
| `document_id` | Unique identifier for the source document or generated text |
| `record_id` | Unique identifier for the individual text record |
| `text` | The actual text used for modelling |
| `source_type` | Whether the record is real-world or synthetic |
| `source_title` | Title of the source, where applicable |
| `source_url` | Source URL, where applicable |
| `license` | Licence or reuse terms, where known |
| `topic` | Main coastal-science topic |
| `collection_date` | Date the record was collected or created |
| `quality_status` | Review status of the record |

Not every field will be applicable to every record. Unknown metadata must not be invented.

## 5. Data Quality and Cleaning

Before training, the dataset will be checked for:

- Empty records and corrupted text
- Duplicate and near-duplicate passages
- Irrelevant content and extraction errors
- Broken character encoding and unwanted formatting
- Unsupported scientific claims and misleading synthetic examples
- Missing or inconsistent source metadata
- Copyright, licence, and attribution issues

Cleaning must preserve scientific meaning, including units, equations, technical terms, and meaningful numerical values.

## 6. Dataset Splitting

The dataset will be separated into training, validation, and test sets.

- **Training set:** used to update model parameters.
- **Validation set:** used to monitor training and compare configurations.
- **Test set:** reserved for final evaluation.

Where possible, records derived from the same source document will remain in the same split. This reduces the risk of nearly identical passages appearing in both training and evaluation data.

The split proportions will be selected after the initial corpus size and source structure are known.

## 7. Tokenization Requirements

The corpus will support the development of a tokenizer suitable for a small language model.

The tokenizer should handle:

- Common coastal-science terminology
- Scientific abbreviations and acronyms
- Numbers and measurement units
- Punctuation and sentence boundaries
- Unfamiliar or compound technical words

The vocabulary size and tokenizer design will be selected based on the size and characteristics of the corpus.

## 8. Evaluation and Limitations

CoastMind will be evaluated using held-out text, training and validation loss, perplexity where appropriate, and qualitative inspection of generated text.

Evaluation will examine whether the model learns useful language patterns and how its outputs behave on coastal-science prompts.

Low loss or perplexity does not, by itself, establish scientific correctness. Generated statements must not be treated as verified scientific findings without independent evidence.

Because CoastMind is a small educational model, it is expected to have limited knowledge, limited reasoning ability, and a tendency to generate incorrect or unsupported statements.

## 9. Initial Implementation Plan

1. Finalise the dataset specification.
2. Identify and assess suitable text sources.
3. Build a small, documented pilot corpus.
4. Inspect and clean the collected text.
5. Implement and test the tokenizer.
6. Create leakage-aware training, validation, and test splits.
7. Train a baseline language model.
8. Record experiments and evaluate results.

## 10. Open Decisions

The following decisions will be made using evidence from the pilot corpus:

- Initial corpus size
- Text-record length
- Tokenizer type and vocabulary size
- Training, validation, and test proportions
- Balance between real-world and synthetic text
- Which sources can be redistributed with the repository
- Whether additional data storage or versioning tools are needed

These decisions will be documented as the project develops.