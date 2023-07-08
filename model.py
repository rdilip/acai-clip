import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, einsum, rearrange


def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


class AcaiCLIP(nn.Module):
    def __init__(
        self,
        model_A,
        model_B,
        token_dropout: float,
        model_A_output_dim: int,
        dim: int = 256,
        dcl_loss: bool = True,
        filip_similarity_score: bool = True,
    ):
        self.model_A = model_A

        for param in self.model_A.parameters():
            param.requires_grad = False
        self.model_B = model_B

        self.token_dropout = TokenDropout(prob=token_dropout)
        # Used as a projection for LiT
        self.linear = nn.Linear(model_A_output_dim, dim)
        self.dcl_loss = dcl_loss
        self.filip_similarity_score = filip_similarity_score
        self.temperature = nn.Parameter(torch.tensor(0.1))

    # TODO: update to allow for augmented groups. For now, no need.
    # TODO: consider treating attp and attb as different groups.
    # TODO: consider adding an attp token and an attp token (learnable)
    #       then we could treat them as on even footing, and it would be fine.
    def forward(self, batch_A, mask_A, batch_B, mask_B):
        tokens_A, tokens_B = map(self.token_dropout, (batch_A, batch_B))

        hA = self.model_A(tokens_A, mask=mask_A)
        hA = self.linear(hA)
        hA = rearrange(hA, "b t d -> 1 b t d")

        hB = self.model_B(tokens_B, mask=mask_B)
        hB = rearrange(hB, "b t d -> 1 b t d")

        hA = hA / hA.norm(dim=-1, keepdim=True)
        hB = hB / hB.norm(dim=-1, keepdim=True)

        if self.filip_similarity_score:
            logits = filip_similarity_score(hA, hB, self.temperature)
        else:
            logits = einsum(hA, hB, "b1 t d, b2 t d -> b1 b2") / self.temperature

        if self.dcl_loss:
            loss = decoupled_contrastive_loss(logits)
        else:
            loss = default_clip_loss(logits)

        return logits, loss


def default_clip_loss(logits):
    l1 = F.cross_entropy(logits, torch.arange(logits.shape[0]).to(logits.device))
    l2 = F.cross_entropy(logits.T, torch.arange(logits.shape[0]).to(logits.device))
    loss = ((l1 + l2) / 2).mean()
    return loss


def decoupled_contrastive_loss(
    logits: torch.Tensor,  # B, B
):
    exp_logits = torch.exp(logits)
    pos_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    exp_logits = exp_logits.masked_fill(pos_mask, 0)

    exp_logits_denom_A = reduce(exp_logits, "b d -> b", "sum")
    exp_logits_denom_B = reduce(exp_logits, "d b -> b", "sum")
    LA = exp_logits.diagonal() / exp_logits_denom_A
    LB = exp_logits.diagonal() / exp_logits_denom_B

    return 0.5 * (-LA.log() - LB.log()).mean()


class TokenDropout(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        assert 0 <= prob <= 1
        self.prob = prob

    # For now, we'll just dropout tokens without too much extra thought.
    # We may eventually want to first drop padded tokens, then drop the meaningful
    # ones, but that's an added complication
    def forward(self, x, keep_all=False):
        if not self.training or self.prob == 0.0 or keep_all:
            return x

        B, T, D = x.shape
        num_tokens_keep = max(1, int(T * (1 - self.prob)))

        batch_indices = torch.arange(B, device=x.device)[..., None]
        token_indices_keep = (
            torch.randn(B, T, device=x.device).topk(num_tokens_keep, dim=-1).indices
        )

        return x[batch_indices, token_indices_keep], token_indices_keep


def masked_mean(t, mask, dim=1, eps=1e-6):
    t = t.masked_fill(~mask, 0.0)
    numer = t.sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=eps)
    return numer / denom


def filip_similarity_score(
    hA: torch.Tensor,
    hB: torch.Tensor,
    maskA: torch.Tensor,
    maskB: torch.Tensor,
    temperature: torch.float,
):
    """Computes a filip similarity score (described in
    https://arxiv.org/pdf/2111.07783.pdf) for modalities A and B.
    # recombinase clusters, though I suspect this may only be true on the protein side.
    hA:     group, batch, time, dim
    hB:     group, batch, time, dim
    maskA:  group, batch, time
    maskB:  group, batch, time
    """
    sim_scores = einsum(hA, hB, "m bA tA d, n bB tB d -> m n bA bB tA tB") / temperature
    maskA = rearrange(maskA, "m bA tA -> m 1 bA 1 tA 1")
    maskB = rearrange(maskB, "n bB tB -> 1 n 1 bB 1 tB")
    combined_mask = maskA * maskB

    # Mask out all padding tokens for both modalities
    sim_scores_masked = sim_scores.masked_fill(
        ~combined_mask, torch.finfo(sim_scores.dtype).min
    )

    sim_scores_A = reduce(sim_scores_masked, "... tA tB -> ... tA", "max")
    sim_scores_B = reduce(sim_scores_masked, "... tA tB -> ... tB", "max")

    sim_scores_A = masked_mean(
        sim_scores_A, rearrange(maskA, "... tA 1 -> ... tA"), dim=-1
    )

    sim_scores_B = masked_mean(
        sim_scores_B, rearrange(maskB, "... 1 tB -> ... tB"), dim=-1
    )

    return sim_scores_A, sim_scores_B
