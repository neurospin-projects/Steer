# STEER

Official implementation of **STEER: Steering Shared and Modality-Specific
Information in Multimodal Self-Supervised Learning**.

STEER learns a preference-conditioned family of multimodal representations
within a single pretrained model. A preference

<p align="center">
  <b>λ = (λ<sub>R</sub>, λ<sub>U1</sub>, λ<sub>U2</sub>)</b>
</p>

controls the balance between information shared across modalities and
information specific to each modality.

Instead of committing to a single information profile during pretraining,
STEER allows the operating point to be selected after pretraining according to
the downstream task, without retraining the representation model.
