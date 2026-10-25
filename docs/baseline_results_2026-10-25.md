# Image baseline results — 2026-10-25

The ViT, ConvNeXt, and ResNeXt image baselines completed training on the seed-42 split of the 171-patient cohort. Each used the configured 200-epoch maximum, patience 15, learning rate 1e-4, AMP, and ImageNet-pretrained weights. All stopped early after 18–19 epochs, with their best validation-accuracy checkpoints at epochs 3–4.

The held-out test split contains 31 patients (22 low grade, 9 high grade) and 6,348 slices. Patient-level metrics are primary; slice-level metrics are secondary because slices from one patient are correlated.

| Model | Best val accuracy (epoch) | Image AUROC (95% CI) | Patient AUROC (95% CI) | Patient accuracy |
|---|---:|---:|---:|---:|
| ViT | 0.673 (4) | 0.637 (0.621–0.655) | 0.732 (0.489–0.958) | 0.806 |
| ConvNeXt | 0.695 (3) | 0.583 (0.567–0.598) | 0.657 (0.423–0.881) | 0.710 |
| ResNeXt | 0.646 (3) | 0.609 (0.594–0.624) | 0.722 (0.484–0.929) | 0.645 |

The patient-level confidence intervals are wide because there are only 31 test patients. These results are descriptive and do not motivate a hyperparameter search or accuracy-improvement phase. The early best epochs and rising validation loss are retained as observations about training stability; they do not change the chosen protocol.

Full per-model metrics, predictions, epoch logs, and run configurations are in `results/metrics/baseline_{vit,convnext,resnext}_*` and `results/metrics/baselines_fullcohort_train.log`.
