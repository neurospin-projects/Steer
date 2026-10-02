# STEER

**Preference-conditioned multimodal self-supervised learning for shared and modality-specific information**

STEER learns a family of multimodal representations within a single pretrained
model, allowing different downstream tasks to select different balances of
shared and modality-specific information without retraining the representation
model.

A preference is defined as

<p align="center">
  <b>λ = (λ<sub>R</sub>, λ<sub>U1</sub>, λ<sub>U2</sub>)</b>
</p>

where **λ<sub>R</sub>** emphasizes information shared across modalities,
while **λ<sub>U1</sub>** and **λ<sub>U2</sub>** emphasize information specific
to modalities 1 and 2, respectively.

Rather than committing to one information profile during pretraining, STEER
learns multiple operating points jointly. Once a downstream task is available,
the pretrained model is frozen and an appropriate operating point is selected
using downstream validation data.

<p align="center">
  <img src="assets/STEER.png" width="900">
</p>

<p align="center">
  <em>
  Overview of STEER. Preference-conditioned encoders learn a family of
  representations spanning different balances of shared and modality-specific
  information. The operating point is selected after pretraining according to
  the downstream task.
  </em>
</p>

---

## Overview

STEER is motivated by multimodal pretraining settings in which future
downstream tasks are not known in advance. Different tasks may rely on
different combinations of information:

- **Redundant information (R):** information available across both modalities.
- **Modality-specific information (U1):** information specific to modality 1.
- **Modality-specific information (U2):** information specific to modality 2.

STEER introduces the preference directly into the modality-specific encoders
through low-rank preference-conditioned adaptations. A single set of model
parameters therefore supports multiple information profiles.

At downstream time:

1. the pretrained representation model is frozen;
2. candidate operating points are evaluated using lightweight probes;
3. a task-specific preference is selected from validation data;
4. the selected operating point is used for final evaluation.

The selected preference provides an operational description of the balance of
shared and modality-specific information used by a task. 

---

## Repository structure

```text
Steer/
│
├── assets/
│   └── STEER.png
│
├── pareto_ssl/
│   ├── benchmark.py
│   ├── datasets.py
│   ├── losses.py
│   ├── networks.py
│   ├── factorcl.py
│   ├── epo.py
│   ├── pareto_config.yaml
│   ├── plot_pareto.py
│   ├── plot_pareto_front_3d.py
│   │
│   ├── trifeature/
│   │   └── generate_corr_variant.py
│   │
│   └── multibench/
│       ├── benchmark_multibench.py
│       ├── probe.py
│       ├── consensus_pick.py
│       ├── task_lambda_probe.py
│       ├── build_mosei_multitask.py
│       ├── mosei_multitask.py
│       ├── multitask_registry.py
│       ├── setup_avmnist.py
│       ├── image_backends.py
│       └── pid_analysis/
│
├── steer_neuro/
│   ├── train_steer_neuro.py
│   ├── train_baselines_neuro.py
│   ├── model.py
│   ├── model_baselines.py
│   ├── encoders.py
│   ├── networks.py
│   ├── losses.py
│   ├── wrap_lora.py
│   ├── paired_dataset.py
│   ├── extract_embeddings.py
│   ├── eval_steer_neuro.py
│   ├── eval_downstream.py
│   ├── select_lambda.py
│   ├── select_lambda_val_test.py
│   ├── probe_baselines.py
│   ├── simplex.py
│   └── diagnostics.py
│
└── README.md
