import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


def default_clip_loss(
    hA,
    hB,
    temperature=0.1,
):
    hA = hA / hA.norm(dim=-1, keepdim=True)
    hB = hB / hB.norm(dim=-1, keepdim=True)

    logits = hA @ hB.T / temperature

    l1 = F.cross_entropy(logits, torch.arange(logits.shape[0]).to(logits.device))
    l2 = F.cross_entropy(logits.T, torch.arange(logits.shape[0]).to(logits.device))
    loss = ((l1 + l2) / 2).mean()
    return loss


def decoupled_contrastive_loss(
    hA: torch.Tensor,  # B, D
    hB: torch.Tensor,  # B, D
    temperature: float = 0.1,
):
    hA = hA / hA.norm(dim=-1, keepdim=True)
    hB = hB / hB.norm(dim=-1, keepdim=True)
    logits = hA @ hB.T / temperature

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

    # Should we do token dropout in addition to or including padded tokens?
    # There is a part that is for efficiency and there is a part that is for
    # learning. If a large part of your dropout involves padded tokens, then
    # you actually aren't making the problem much harder...
    # We could also pad after dropping out lol...
    # you need to encode some 
    def forward(self, x, keep_all=False):
        if not self.training or self.prob == 0. or keep_all:
            return x
        
        B, T, D = x.shape
        num_tokens_keep = max(1, int(T * (1 - self.prob)))

        batch_indices = torch.arange(B, device=x.device)[..., None]
        token_indices_keep = torch.randn(B, T, device=x.device).topk(num_tokens_keep, dim = -1).indices

        return x[batch_indices, token_indices_keep]
