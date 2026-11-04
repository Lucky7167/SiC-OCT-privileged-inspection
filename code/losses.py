
import torch.nn.functional as F

from common import (
    feature_alignment_loss,
    logit_distillation_loss,
    pairwise_rank_loss,
)


def teacher_loss(output, batch, class_counts):
    return (
        F.cross_entropy(output["logits"], batch["z_star"])
        + F.huber_loss(output["q"], batch["q_star"], delta=1.0)
        + F.huber_loss(output["score"], batch["U_star"], delta=1.0)
    )


def student_loss(output, batch, class_counts, temperature=3.0):
    return (
        F.cross_entropy(output["logits"], batch["z_star"])
        + pairwise_rank_loss(output["score"], batch["U_star"])
        + feature_alignment_loss(output["projected_feature"], batch["teacher_feature"])
        + 0.5 * logit_distillation_loss(
            output["logits"], batch["teacher_logits"], temperature
        )
        + F.huber_loss(output["score"], batch["teacher_score"], delta=1.0)
    )


def direct_loss(output, batch, class_counts):
    return (
        F.cross_entropy(output["logits"], batch["z_star"])
        + pairwise_rank_loss(output["score"], batch["U_star"])
    )


def direct_reg_loss(output, batch, class_counts):
    return direct_loss(output, batch, class_counts) + F.huber_loss(
        output["score"], batch["U_star"], delta=1.0
    )
