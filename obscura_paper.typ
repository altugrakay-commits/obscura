#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Physics-Informed Multi-Task Learning for Satellite-Based Industrial Emission Monitoring],
  authors: (
    (
      name: "Altug Remzi Akay, Ph.D.",
      organization: [Bergsskolan - School of Mining and Metallurgy],
      location: [Filipstad, Sweden],
      email: "altug.r.akay@gmail.com",
    ),
  ),
  abstract: [Clouds and sensor limits obscure satellite measurements of industrial emissions, leaving most target values, especially methane, missing from training data. The OBSCURA dataset adds a second hurdle: models must generalize to unseen facilities and later time periods, where purely convolutional approaches overfit. I proposed a late-fusion neural architecture that combines 14-band Sentinel-2 imagery with 29 meteorological and terrain features. To handle missing labels, I introduced a *Masked Mean Squared Error (MSE) loss* that computes gradients only on observed targets, allowing training across all 3,218 samples. Drawing on metallurgy and combustion thermodynamics, I also embedded a *physics-informed penalty* that suppresses impossible predictions, such as intense fire without smoke or dust (AAI). Under grouped cross-validation by facility (`site_key`), my single model achieves a validation masked RMSE of 0.988, which improves to 0.892 with a 5-fold ensemble. Although domain shift degrades raw error on the private leaderboard, the physical constraints reliably flag impossible emission states in an interactive Streamlit dashboard, providing an essential safety net for real-world environmental monitoring.],
  
  index-terms: ("Satellite remote sensing", "Multi-task learning", "Physics-informed neural networks", "Industrial emissions", "Sentinel-2"),
)

= Introduction

Refineries, steelworks, smelters, and petrochemical plants emit major quantities of short-lived climate pollutants, including methane (CH₄), nitrogen dioxide (NO₂), and carbon monoxide (CO) @liebmann2022. Satellites observe these facilities globally, yet cloud cover and retrieval failures frequently leave ground-truth measurements blank. In the OBSCURA competition, CH₄ is missing in roughly 65% of training rows @ru2021. Discarding incomplete rows forfeits more than half the dataset.

Domain shift poses a second challenge. The test set comprises unseen facilities imaged at later dates @ru2021. Purely visual convolutional neural networks (CNNs) memorize local site traits—soil color, sun angle, or building layout—rather than physical dispersion. On new sites, such models often perform worse than predicting the global mean.

Solving both problems requires multi-modal fusion and physical awareness. Unlike target gases, the 29 tabular features (temperature, wind speed, humidity, terrain slope, and spectral indices) appear in every row @ru2021. Fusing these descriptors with 14-band Sentinel-2 imagery supplies the environmental context needed to interpret visual signals. Furthermore, combustion thermodynamics and Gaussian plume physics forbid certain emission states: intense fire cannot occur without particulate matter, nor can dense smoke persist under high winds. I introduce a custom loss function that penalizes predictions violating these laws, drawing on recent advances in Physics-Informed Neural Networks @raissi2019.

This paper makes three contributions:
1. A late-fusion neural architecture that jointly processes image chips and tabular features.
2. A Masked MSE loss that trains on incomplete labels without discarding data.
3. A physics-informed penalty that enforces thermodynamic consistency and powers a real-time violation dashboard.

= Methodology

I framed the OBSCURA task as a multi-modal regression under severe label missingness and physical boundary constraints.

== Data Preprocessing and Pipeline

The training set contains 3,218 samples, each linked to a geographic site (`site_key`). Every sample pairs a $384 times 384$ Sentinel-2 image chip (14 spectral bands) with 29 meteorological and terrain features.

Reading 14-band `.tif` files from the disk bottlenecks GPU training. I built a custom PyTorch `Dataset` that normalizes images to $[0, 1]$ during the first epoch and caches them in system RAM (64 GB). Caching cut subsequent epoch times from 200 seconds to 25 seconds.

I imputed missing tabular entries with zero and standardized all features using `StandardScaler`, fitting mean and variance strictly on each training fold to prevent data leakage. To mirror the unseen-site test split, I evaluated all models with `GroupKFold` cross-validation grouped by (`site_key`).

== Neural Architecture

My late-fusion CNN processesed spatial and tabular inputs in separate branches before combining them:\

=== Image Branch (CNN)

Three blocks, each containing a $3 times 3$ convolution, Batch Normalization, ReLU activation, and $2 times 2$ max pooling, expanded channels from 14 → 32 → 64 → 128. Adaptive average pooling then collapses spatial dimensions to $1 times 1$, yielding a 128-dimensional feature vector.\
\
=== Tabular Branch (MLP)

A Multi-Layer Perceptron projected the 29 standardized features through a Linear layer (29 → 64), ReLU activation, Dropout (p = 0.2), and a second Linear layer (64 → 32).\

=== Late Fusion

Concatenating both branches produced a 160-dimensional vector. A fully connected head (Linear 160 → 64, ReLU, Linear 64 → 5) predicts CH₄, NO₂, CO, AAI, and FIRMS.\

== Masked Loss Function

Standard MSE fails when targets contain `NaN` values. Let $y_i$ denote the ground truth for target $i$, $hat(y)_i$ the prediction, and $M_i in {0,1}$ a binary mask equal to 1 when $y_i$ is observed. I computed Masked MSE strictly over observed targets:

$ cal(L)_"MSE" = frac(1, sum_i M_i) sum_(i=1)^5 M_i (y_i - hat(y)_i)^2 $

This formulation trains on all 3,218 samples without discarding  incomplete rows, expanding the effective training set.

== Physics-Informed Penalty

Unconstrained models can predict high Fire Intensity (FIRMS) alongside zero Smoke/Dust (AAI), violating combustion thermodynamics. I added a penalty $cal(L)_"physics"$ that triggers whenever predicted fire intensity exceeds 1.0 while predicted smoke falls below zero:

$ cal(L)_"physics" = "mean" ( "ReLU" (0 - hat(y)_"AAI") dot bb(1)(hat(y)_"FIRMS" > 1.0) ) $

The total objective sums the regression loss and weighted penalty:

$ cal(L)_"total" = cal(L)_"MSE" + lambda cal(L)_"physics" $

where $lambda = 0.01$ balances data fit against physical laws.

== Training Protocol

I trained the model on an NVIDIA GeForce RTX 4060 Laptop GPU using AdamW (initial learning rate $1 times 10^(-3)$, weighted decay $1 times 10^(-4)$). A `ReduceLROnPlateau` scheduler halved the learning rate after 5 stagnant validation epochs, and early stopping halted training after 4 epochs without improvement. Random horizontal flips, vertical flips, and 90-degree rotations regularized the image branch.

= Experiments and Results

I compared my PyTorch architecture against an XGBoost baseline and a naive mean baseline using masked Root Mean Squared Error (RMSE) on local grouped validation and the Kaggle public leaderboard.

== Local Cross-Validation Results

I evaluated models locally with 5-fold `GroupKFold` cross-validation partitioned by `site_key`, ensuring no training facility appears in validation folds.

Table I summarizes validation performance. Trained solely on tabular features, XGBoost achieved a masked RMSE of 1.052. My late-fusion model lowered validation RMSE to 0.988 for a single model and 0.892 (MSE 0.7956) for the 5-fold ensemble.

#figure(
  table(
    columns: 3,
    align: center,
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Features Used*], [*Validation RMSE*]),
    table.hline(),
    [Naive Mean Baseline], [None], [1.145],
    [XGBoost Baseline], [Tabular Only], [1.052],
    [PyTorch Single Model], [Images + Tabular], [0.988],
    [*PyTorch 5-Fold Ensemble*], [*Images + Tabular*], [*0.892*],
    table.hline(),
  ),
  caption: [Local Grouped Validation Performance (Masked RMSE).\ Lower is better.],
) <tab-results>

Improving RMSE from 1.052 to 0.892 shows that multi-spectral imagery adds spatial context beyond meteorological features alone. Ensembling across five folds also lowered cross-fold variance, confirming that geometric augmentations regularize the CNN.

== Kaggle Public Leaderboard and the Generalization Gap

On the Kaggle public leaderboard, the naive mean baseline (predicting zero for normalized targets) scored 1.145, whereas my 5-fold ensemble scores 1.186.

This gap illustrates domain shift. Convolutional filters memorized site-specific visual textures—soil color, building layout, and seasonal satellite angles—and extrapolated poorly on unseen test facilities. With only 81 unique training sites and no satellite foundation-model pretraining (such as ImageNet or BigEarthNet), purely visual CNNs readily overfit training geographies.

== Physics-Informed Validation and Case Studies

Despite higher raw RMSE under domain shift, my physical constraints reliably catch impossible outputs. I tested the model on the two most extreme training samples:

#figure(
  image("data/image_test/demo_fire.png", width: 80%),
  caption: [Combustion Violation: The model predicts high fire intensity (1.828) but low smoke (0.042). The thermodynamic constraint successfully flags this as physically inconsistent.],
) <fig-fire>

*Case 1: Combustion Thermodynamics.* On the highest-fire sample, the model predicted high fire intensity (1.828) but near-zero smoke (0.042). As shown in @fig-fire, the Streamlit dashboard immediately flagged a Combustion Violation. Rather than silently passing an impossible prediction, the physics layer alerts the operator.

#figure(
  image("data/image_test/demo_smoke.png", width: 80%),
  caption: [Dispersion Violation: Under high wind speed (20 m/s), the model predicts heavy smoke (2.129). The atmospheric dispersion constraint successfully flags this as physically inconsistent.],
) <fig-smoke>

*Case 2: Atmospheric Dispersion.* On the highest-smoke sample under a simulated 20 m/s wind speed, the model predicted heavy smoke (2.129). As shown in @fig-smoke,  the dashboard flagged a Dispersion Violation because dense particulates cannot persist at that wind speed under Gaussian plume dispersion models.

These case studies show that physical guardrails provide an essential safety net. In operational monitoring, catching physically impossible readings matters far more than shaving decimal places off a global error metric.

= Discussion

The findings highlight both the limits of small-sample deep vision models and the value of physical guardrails.

== The Domain Shift Challenge
Eighty-one training sites cannot teach a randomly initialized CNN to learn site-invariant spatial features. Without pretraining on global remote-sensing corpora, visual weights misalign on unseen geographies.

== The Value of Physics-Informed Regularization
Embedding combustion thermodynamics and Gaussian dispersion rules creates a system that recognizes its own failures. When extreme inputs trigger impossible outputs, the Streamlit dashboard immediately flags Combustion and Dispersion Violations. Regardless of visual domain shift, the pipeline never silently reports thermodynamically impossible emissions.

== Limitations and Future Work
Future work should replace my custom CNN with a pretrained Vision Transformer (ViT) or ResNet fine-tuned on Sentinel-2 bands via self-supervised learning. Adding a temporal model, such as a ConvLSTM, across consecutive satellite passes would further strengthen plume dispersion modeling.

= Conclusion

I presented a physics-informed, multi-modal neural network for industrial emission monitoring. Masked MSE enables training on incomplete targets, late fusion integrates meteorological context, and metallurgical thermodynamics catch physically impossible predictions. Although limited site diversity constrains raw error on unseen facilities, pairing domain physics with deep learning provides a reliable early-warning safety net for environmental monitoring.

#bibliography("reference.bib")