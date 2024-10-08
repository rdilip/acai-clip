import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, einsum, rearrange


class AcaiCLIP(nn.Module):
    def __init__(
        self,
        model_A,
        model_B,
        token_dropout: float,
        dcl_loss: bool = True,
        model_A_output_dim: int= 1280,
        n_embd: int = 128,
        filip_similarity_score: bool = True,
    ):
        self.model_A = model_A
        self.model_B = model_B

        self.token_dropout = TokenDropout(prob=token_dropout)
        # Used as a projection for LiT
        self.proj = nn.Linear(model_A_output_dim, n_embd)
        self.dcl_loss = dcl_loss
        self.filip_similarity_score = filip_similarity_score
        self.temperature = nn.Parameter(torch.tensor(0.1))

    # TODO: update to allow for augmented groups. For now, no need.
    # TODO: consider treating attp and attb as different groups.
    # TODO: consider adding an attp token and an attp token (learnable)
    #       then we could treat them as on even footing, and it would be fine.
    # NOTE: If you do go this route, you should be careful to distinguish
    #       between SSL, SLIP etc. and two groups from the same batch.
    def forward(self, batch_A, mask_A, batch_B, mask_B):
        tokens_A, tokens_B = map(self.token_dropout, (batch_A, batch_B))

        hA = self.model_A(tokens_A, mask=mask_A)
        hA = self.proj(hA)
        hA = rearrange(hA, "b t d -> 1 b t d")

        hB = self.model_B(tokens_B, mask=mask_B)
        hB = rearrange(hB, "b t d -> 1 b t d")

        hA = hA / hA.norm(dim=-1, keepdim=True)
        hB = hB / hB.norm(dim=-1, keepdim=True)

        if self.filip_similarity_score:
            logits = filip_similarity_score(hA, hB, self.temperature)
        else:
            hA = reduce(hA, "1 b t d -> b d", "mean")
            hB = reduce(hB, "1 b t d -> b d", "mean")
            logits = einsum(hA, hB, "b1 d, b2 d -> b1 b2") / self.temperature

        if self.dcl_loss:
            loss_A, loss_B = decoupled_contrastive_loss(logits)
        else:
            loss_A, loss_B = default_clip_loss(logits)

        loss = (loss_A + loss_B).mean() / 2

        return logits, loss

def contrastive_loss(logits: torch.Tensor, use_dcl: bool = False):
    exp_logits = logits.exp()
    loss_numerator = exp_logits.diagonal()

    if use_dcl:
        pos_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        exp_logits = exp_logits.masked_fill(pos_mask, 0)
    
    loss_denominator = reduce(exp_logits, "b t -> b", "sum")
    return (-loss_numerator.log() + loss_denominator.log()).mean()


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

def mean_average_similarity_score(
    hA: torch.Tensor, # B, T1, D
    hB: torch.Tensor, # B, T2, D
    maskA: torch.Tensor, # B, T1
    maskB: torch.Tensor, # B, T2
    temperature: torch.float
):
    hA = reduce(hA * maskA[..., None], "b t d -> b d", "mean")
    hB = reduce(hB * maskB[..., None], "b t d -> b d", "mean")
    logits = einsum(hA, hB, "b1 d, b2 d -> b1 b2") / temperature
    return logits

def filip_similarity_score(
    hA: torch.Tensor,
    hB: torch.Tensor,
    maskA: torch.Tensor,
    maskB: torch.Tensor,
    temperature: torch.float,
    include_group: bool = False
):
    """Computes a filip similarity score (described in
    https://arxiv.org/pdf/2111.07783.pdf) for modalities A and B.
    # recombinase clusters, though I suspect this may only be true on the protein side.
    hA:     group, batch, time, dim
    hB:     group, batch, time, dim
    maskA:  group, batch, time
    maskB:  group, batch, time
    """
    # Don't do separate cases for each of these: einops will throw an error if you include
    # a group when you said you don't want to, and this is by design
    if not include_group:
        hA = rearrange(hA, "b t d -> 1 b t d")
        hB = rearrange(hB, "b t d -> 1 b t d")
        maskA = rearrange(maskA, "b t -> 1 b t")
        maskB = rearrange(maskB, "b t -> 1 b t")

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

    if not include_group:
        sim_scores_A, sim_scores_B = map(lambda s: rearrange(s, "1 1 b t -> b t"), (sim_scores_A, sim_scores_B))

    return sim_scores_A, sim_scores_B
