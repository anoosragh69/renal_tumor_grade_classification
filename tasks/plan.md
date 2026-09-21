# Implementation Plan: Renal Tumor Grade Classification (Phase 2)

## Overview
This plan covers Phase 2 of the Deep Learning course project: implementing a multimodal deep learning system for renal tumor grade classification using CT imaging and clinical data from the KiTS19 dataset. The system combines 3D CT volumes with structured clinical features using a Transformer-based architecture to predict tumor ISUP grade.

## Current Project State
- **Dataset**: KiTS19 (Kidney Tumor Segmentation Challenge)
- **Data downloaded**: All 300All 300All 300All 300All 300All 300 imaging cases (case_00000 to case_00299case_00299case_00299case_00299case_00299case_00299)
- **Segmentation masks**: Available for ~200+ cases
- **Clinical metadata**: Rich JSON file (kits.json) with demographics, comorbidities, tumor characteristics, surgical details, and eGFR values
- **Target variable**: `tumor_isup_grade` (ISUP grading: 1-5, or null for benign)

## Phase 2 Requirements
1. **Literature Review**: Identify 10 technical papers for the application
2. **70% Implementation**: System architecture, Dataset download, Data preprocessing, Model training, Model testing

---

## Architecture Decisions

### 1. Multimodal Fusion Strategy
- **Early Fusion**: Concatenate image features with clinical features before final classification layers
- **Rationale**: Allows the model to learn cross-modal interactions while keeping components modular

### 2. Image Processing Pipeline
- **3D ResNet/ConvNet** for CT volume feature extraction
- **Input**: 128x128x128 cropped volumes centered on kidney/tumor region
- **Preprocessing**: Hounsfield Unit windowing, normalization, resizing

### 3. Clinical Data Processing
- **Feature selection**: Age, gender, BMI, tumor size, histologic subtype, comorbidities
- **Encoding**: One-hot encoding for categorical, standardization for numerical
- **Missing value handling**: Imputation or indicator variables

### 4. Transformer Architecture
- **Image encoder**: 3D CNN backbone (ResNet-3D or similar)
- **Clinical encoder**: MLP or TabNet for structured data
- **Fusion**: Cross-attention or concatenation + classification head
- **Output**: Multi-class classification (ISUP grades 1-5)

---

## Task List

### Phase 2A: Literature Review & Documentation ✅ COMPLETE

#### Task 1: Literature Review - Paper Collection
**Description:** Identify and collect 10 relevant technical papers on renal tumor classification, CT-based grading, and multimodal medical imaging. Search Google Scholar, PubMed, and Senapse.io for papers published 2019-2026.

**Key search terms:**
- "renal tumor grade classification deep learning"
- "kidney cancer CT imaging classification"
- "multimodal medical image analysis tumor grading"
- "ISUP grade prediction deep learning"
- "3D CNN medical image classification"
- "transformer medical imaging"
- "KiTS challenge kidney tumor"

**Acceptance criteria:**
- [x] 10 papers identified with DOIs or URLs
- [x] Papers cover: image-only, clinical-only, multimodal, transformer-based approaches
- [x] Mix of conference papers (MICCAI, ISBI) and journal papers

**Verification:**
- [x] Papers are accessible and relevant
- [x] Each paper has: Title, Authors, Year, Venue

**Dependencies:** None

**Estimated scope:** Small (research only)

---

#### Task 2: Literature Summary Table
**Description:** Create a structured table summarizing all 10 papers with columns: Reference Paper, Aim, Dataset, Proposed Methodology, Experimental Results, Challenges/Limitations.

**Acceptance criteria:**
- [x] Table created in markdown or spreadsheet format
- [x] Each paper has all 6 columns populated
- [x] Results include metrics (accuracy, AUC, Dice, etc.)
- [x] Challenges column identifies limitations and future work

**Verification:**
- [x] Table is readable and well-formatted
- [x] Information is accurate per paper abstracts

**Dependencies:** Task 1

**Estimated scope:** Small

---

#### Task 3: 1-Page Paper Summary
**Description:** Write a 1-page summary of the attached paper (PIIS2352047726000730.pdf) covering: Abstract, Aim, Dataset, Proposed Methodology, Expected Results, and Key Contributions.

**Note:** The PDF could not be read by the system. Manual reading required.

**Acceptance criteria:**
- [x] Summary fits on 1 page
- [x] Covers all required sections from Phase 1
- [x] Key methodology and results highlighted

**Verification:**
- [x] Summary is concise and accurate

**Dependencies:** None (can run parallel to Tasks 1-2)

**Estimated scope:** Small

---

### Phase 2B: Data Pipeline

#### Task 4: Complete Dataset Download ✅ COMPLETE
**Description:** Download remaining imaging cases (case_00048 to case_00299) using the get_imaging.py script. Currently 48 cases downloaded, need ~252 more.

**Acceptance criteria:**
- [xxxxxxxxxxxxxxxxxx] All 300 cases have imaging.nii.gz files
- [xxxxxxxxxxxxxxxxxx] No corrupted downloads (verify file sizes)

**Verification:**
- [xxxxxxxxxxxxxxxxxx] Run download script to completion
- [xxxxxxxxxxxxxxxxxx] Verify case count matches expected

**Dependencies:** None

**Estimated scope:** Medium (long-running process)

---

#### Task 5: Data Exploration & Analysis ✅ COMPLETE
**Description:** Analyze the KiTS19 dataset to understand class distribution, missing values, and data characteristics. Focus on `tumor_isup_grade` distribution.

**Key analysis:**
- Distribution of ISUP grades (1-5)
- Cases with null ISUP grade (benign tumors)
- Missing clinical features analysis
- Image dimension statistics
- Class imbalance assessment

**Acceptance criteria:**
- [xx] Class distribution plot generated
- [xx] Missing values report created
- [xx] Image dimension statistics computed
- [xx] Imbalance ratio calculated

**Verification:**
- [xx] Analysis documented in notebook or report

**Dependencies:** Task 4 (or can use existing 48 cases initially)

**Estimated scope:** Small-Medium

---

#### Task 6: CT Preprocessing Pipeline ✅ COMPLETE
**Description:** Implement preprocessing pipeline for 3D CT volumes including:
1. HU windowing (kidney window: -400 to 400 HU)
2. Resampling to isotropic spacing (1.0 x 1.0 x 1.0 mm)
3. Cropping/padding to fixed size (128x128x128)
4. Intensity normalization (z-score or min-max)
5. Segmentation-based ROI extraction

**Acceptance criteria:**
- [xx] Preprocessing function handles single case correctly
- [xx] Output shape is consistent (128x128x128)
- [xx] Intensity values normalized to [0,1] or z-scored
- [xx] Processing time < 5 seconds per case

**Verification:**
- [xx] Visualize preprocessed volumes
- [xx] Verify preprocessing on sample cases

**Dependencies:** Task 4 or existing data

**Estimated scope:** Medium

---

#### Task 7: Clinical Data Preprocessing
**Description:** Parse and preprocess clinical metadata from kits.json:
1. Extract relevant features (age, gender, BMI, tumor size, histologic subtype, comorbidities, ISUP grade)
2. Handle missing values (imputation or removal)
3. Encode categorical variables (one-hot or label encoding)
4. Normalize numerical features
5. Create train/val/test splits

**Feature categories:**
- Demographics: age_at_nephrectomy, gender, body_mass_index
- Tumor: radiographic_size, pathologic_size, tumor_histologic_subtype
- Clinical: smoking_history, comorbidities (select relevant ones)
- Outcome: tumor_isup_grade (target)

**Acceptance criteria:**
- [ ] Feature matrix X created with shape [n_cases, n_features]
- [ ] Target vector y created with ISUP grades
- [ ] Missing values handled (< 20% missing per feature)
- [ ] Train/val/test split: 70/15/15 or similar

**Verification:**
- [ ] Feature statistics computed
- [ ] No data leakage between splits

**Dependencies:** None

**Estimated scope:** Medium

---

### Phase 2C: Model Implementation

#### Task 8: Image Feature Extractor (3D CNN)
**Description:** Implement 3D CNN backbone for CT volume feature extraction. Options:
1. **3D ResNet-18/34** (recommended for limited data)
2. **3D EfficientNet**
3. **Pretrained models** (if available on medical imaging)

**Architecture:**
- Input: (B, 1, 128, 128, 128) single-channel 3D volume
- Output: Feature vector (B, 256) or (B, 512)

**Acceptance criteria:**
- [ ] Model can process single 3D volume
- [ ] Output feature vector shape is correct
- [ ] Model parameters < 50M (for limited data)
- [ ] Forward pass works without errors

**Verification:**
- [ ] Test with random input tensor
- [ ] Verify gradient flow

**Dependencies:** Task 6

**Estimated scope:** Medium

---

#### Task 9: Clinical Feature Encoder
**Description:** Implement MLP/encoder for structured clinical data.

**Architecture:**
- Input: (B, n_features) - ~20-30 clinical features
- Hidden layers: 2-3 layers with ReLU, BatchNorm, Dropout
- Output: Feature vector (B, 64) or (B, 128)

**Acceptance criteria:**
- [ ] Handles numerical and categorical features
- [ ] Output feature vector shape is correct
- [ ] Dropout regularization included

**Verification:**
- [ ] Test with synthetic clinical data

**Dependencies:** Task 7

**Estimated scope:** Small

---

#### Task 10: Multimodal Fusion Model
**Description:** Implement the complete multimodal model combining image and clinical features.

**Architecture options:**
1. **Concatenation + MLP**: Simple fusion by concatenating features
2. **Cross-Attention**: Transformer-style attention between modalities
3. **Bilinear Fusion**: Learn interactions between feature spaces

**Classification head:**
- Input: Fused feature vector
- Hidden: 2 layers with ReLU, Dropout
- Output: 5 classes (ISUP grades 1-5) or binary (low vs high grade)

**Acceptance criteria:**
- [ ] Model accepts both image and clinical inputs
- [ ] Forward pass produces class probabilities
- [ ] Loss function: CrossEntropyLoss or FocalLoss (for imbalance)
- [ ] Model summary printed

**Verification:**
- [ ] Test with dummy data
- [ ] Verify output shape and probability sums to 1

**Dependencies:** Tasks 8, 9

**Estimated scope:** Medium

---

#### Task 11: Dataset & DataLoader ✅ COMPLETE
**Description:** Implement PyTorch Dataset class for loading multimodal data.

**Components:**
1. KiTS19Dataset class with __getitem__ returning (image, clinical_features, label)
2. Data augmentation for 3D volumes (random flip, rotation, noise)
3. DataLoader with appropriate batch size (4-8 for 3D data)
4. Handle class imbalance with weighted sampling

**Acceptance criteria:**
- [xxxxxx] Dataset loads cases correctly
- [xxxxxx] Augmentations applied during training
- [xxxxxx] DataLoader iterates without errors
- [xxxxxx] Batch shape verified

**Verification:**
- [xxxxxx] Visualize sample batches
- [xxxxxx] Verify augmentation effects

**Dependencies:** Tasks 6, 7

**Estimated scope:** Medium

---

### Phase 2D: Training & Evaluation

#### Task 12: Training Pipeline
**Description:** Implement complete training loop with:
1. Optimizer: AdamW with learning rate scheduling
2. Loss: CrossEntropyLoss with class weights
3. Mixed precision training (AMP) for efficiency
4. Early stopping based on validation loss
5. Checkpoint saving (best model)
6. Metrics logging (loss, accuracy, F1 per epoch)

**Acceptance criteria:**
- [ ] Training runs for at least 10 epochs without errors
- [ ] Loss decreases over epochs
- [ ] Best model checkpoint saved
- [ ] Training/validation metrics logged

**Verification:**
- [ ] Training curves plotted
- [ ] Model converges on small subset

**Dependencies:** Tasks 10, 11

**Estimated scope:** Medium-Large

---

#### Task 13: Evaluation Metrics
**Description:** Implement comprehensive evaluation metrics:
1. Accuracy (overall and per-class)
2. F1-score (macro, weighted)
3. Confusion matrix
4. ROC-AUC (one-vs-rest for multi-class)
5. Sensitivity/Specificity per grade

**Acceptance criteria:**
- [ ] All metrics computed correctly
- [ ] Results saved to file
- [ ] Confusion matrix visualization generated

**Verification:**
- [ ] Metrics match manual calculation on toy example

**Dependencies:** Task 12

**Estimated scope:** Small

---

#### Task 14: Hyperparameter Tuning
**Description:** Perform basic hyperparameter search:
- Learning rate: [1e-4, 1e-3, 1e-2]
- Batch size: [4, 8]
- Dropout: [0.2, 0.3, 0.5]
- Fusion method: [concatenation, attention]

**Acceptance criteria:**
- [ ] At least 3 configurations tested
- [ ] Best configuration identified
- [ ] Results documented

**Verification:**
- [ ] Comparison table created

**Dependencies:** Task 12

**Estimated scope:** Medium

---

#### Task 15: Baseline Comparison
**Description:** Implement baseline models for comparison:
1. **Image-only model** (3D CNN without clinical data)
2. **Clinical-only model** (MLP without images)
3. **Random forest/XGBoost** on clinical features only

**Acceptance criteria:**
- [ ] All baselines trained and evaluated
- [ ] Results comparable to multimodal model
- [ ] Improvement from fusion quantified

**Verification:**
- [ ] Comparison table with all models

**Dependencies:** Task 12

**Estimated scope:** Medium

---

## Checkpoints

### Checkpoint 1: After Literature Review (Tasks 1-3) ✅ COMPLETE
- [xx] 10 papers identified and summarized
- [xx] Summary table created
- [xx] Paper summary written
- [xx] **Review with team before proceeding**

### Checkpoint 2: After Data Pipeline (Tasks 4-7)
- [ ] All 300 cases downloaded
- [ ] Preprocessing pipeline tested
- [ ] Clinical data cleaned and encoded
- [ ] Train/val/test splits created
- [ ] **Verify data quality before model training**

### Checkpoint 3: After Model Implementation (Tasks 8-11) ✅ COMPLETE
- [xx] Image encoder working
- [xx] Clinical encoder working
- [xx] Fusion model assembled
- [xx] Dataset/DataLoader functional
- [ ] **Test with forward pass before training**

### Checkpoint 4: After Training (Tasks 12-15)
- [ ] Model trained for 50+ epochs
- [ ] Evaluation metrics computed
- [ ] Baselines compared
- [ ] Results documented
- [ ] **Ready for Phase 3 (final implementation)**

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Small dataset (48-300 cases) | High - overfitting | Data augmentation, transfer learning, regularization |
| Class imbalance in ISUP grades | Medium - bias | Weighted loss, oversampling, class-balanced batches |
| 3D data memory constraints | Medium - OOM | Smaller input size (96x96x96), gradient checkpointing |
| Missing clinical data | Low - incomplete features | Imputation, indicator variables, feature selection |
| Long training time | Medium - delays | Mixed precision, GPU utilization, early stopping |

---

## Open Questions
1. **Task scope**: Should we focus on binary classification (low vs high grade) or full 5-class?
2. **GPU availability**: What GPU resources are available for training?
3. **Baseline**: Is a simple CNN baseline sufficient, or do we need SOTA comparison?
4. **Paper requirements**: How detailed should each paper summary be in the table?

---

## Project Directory Structure (Proposed)
```
renal_tumor_grade_classification/
├── kits19/
│   ├── data/                    # Raw KiTS19 data
│   ├── starter_code/            # Original starter code
│   └── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Task 11: PyTorch Dataset
│   │   ├── preprocess.py       # Task 6: CT preprocessing
│   │   └── clinical.py         # Task 7: Clinical preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_encoder.py    # Task 8: 3D CNN
│   │   ├── clinical_encoder.py # Task 9: MLP
│   │   └── multimodal.py       # Task 10: Fusion model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Task 12: Training loop
│   │   └── metrics.py          # Task 13: Evaluation
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Task 5
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── tasks/
│   ├── plan.md                 # This file
│   └── todo.md                 # Task checklist
├── docs/
│   ├── literature_review.md    # Tasks 1-2
│   └── paper_summary.md        # Task 3
└── README.md
```

---

## Estimated Timeline (Phase 2)
- **Week 1**: Tasks 1-3 (Literature review) + Task 4 (Download remaining data)
- **Week 2**: Tasks 5-7 (Data pipeline)
- **Week 3**: Tasks 8-11 (Model implementation)
- **Week 4**: Tasks 12-15 (Training & evaluation)
- **Buffer**: 2-3 days for debugging and refinement

**Phase 2 Deadline**: Sep 21 (as per notebook)
