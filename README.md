# Learning when to scan: OCT-privileged selective acquisition for resource-efficient silicon carbide wafer inspection

## Introduction

This repository contains code for a robot-guided microscopy–OCT framework that uses OCT as privileged information during training and selectively acquires depth-resolved measurements during inspection.

## Abstract

Industrial inspection must balance rapid screening against information-rich characterization because no single modality provides both large-area coverage and comprehensive physical evidence. Silicon carbide (SiC) wafer inspection illustrates this trade-off. Optical microscopy rapidly localizes candidate defects, but surface morphology does not uniquely reveal their three-dimensional structure. Optical coherence tomography (OCT) provides complementary depth-resolved scattering information at higher acquisition cost. Here, we present a robot-guided microscopy–OCT framework that learns which microscopy-localized candidates require depth-resolved verification. During training, OCT volumes provide privileged information to a microscopy-only model that predicts defect categories and an ordinal subsurface-likelihood score for referral. At deployment, the strategy invokes OCT for intermediate-score candidates and resolves low- and high-score cases directly from microscopy. In leave-one-wafer-out evaluations of five wafers from distinct manufacturing batches, the method retained all 305 OCT-identified subsurface defects within the microscopy-defined candidate pools. It used 27.4% of the OCT acquisition time required by candidate-guided scanning and 9.8% of that required by exhaustive wafer scanning. These findings show how information-rich modalities can guide training of policies that determine when those modalities are needed at deployment. This approach supports resource-efficient multimodal inspection through selective acquisition of complementary physical evidence.

## Updates

This repository will be updated progressively.

```text
code/
tests/
requirements.txt
LICENSE
```

```bash
pip install -r requirements.txt
python -m pytest -q
python code/run_all.py --train --output outputs/lowo
python code/run_all.py --train --output outputs/lowo_all_seeds --seeds 0 1 2 3 4
```
