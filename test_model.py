from model import (
    TokenDropout,
    filip_similarity_score,
)
import torch
from einops import rearrange
from itertools import product


def test_decoupled_contrastive_loss():
    # To run this test, comment out the masking in loss.py
    batch_size = 16
    embd_dim = 128
    hA, hB = torch.randn(batch_size, embd_dim), torch.randn(batch_size, embd_dim)
    # assert torch.isclose(default_clip_loss(hA, hB), decoupled_contrastive_loss(hA, hB))


def test_token_dropout():
    B, T, D = 16, 128, 512
    x = torch.randn(B, T, D)
    token_dropout, _ = TokenDropout(0.5)
    x = token_dropout(x)
    assert x.shape == (B, T // 2, D)


def test_similarity_masking():
    torch.manual_seed(0)
    m, n = 3, 1
    bA, bB = 32, 64
    Ta, Tb = 128, 256

    maskA = torch.rand(m, bA, Ta) > 0.5
    maskB = torch.rand(n, bB, Tb) > 0.5

    maskA = rearrange(maskA, "m bA tA -> m 1 bA 1 tA 1")
    maskB = rearrange(maskB, "n bB tB -> 1 n 1 bB 1 tB")

    combined = maskA * maskB

    for group in range(m):
        for batch in range(bA):
            for token in range(Ta):
                if not maskA[group, 0, batch, 0, token, 0].item():
                    assert torch.all(~combined[group, :, batch, :, token, :]).item()


def test_similarity_score_single_batch():
    """Compare similarity scoring from filip to loopy version."""
    torch.manual_seed(0)
    Ta, Tb = 128, 256  # sequence steps
    d = 50  # embedding dimension

    hA = torch.randn(Ta, d)
    hB = torch.randn(Tb, d)
    maskA = torch.rand(Ta) > 0.2
    maskB = torch.rand(Tb) > 0.2
    temperature = torch.randn(1).item()

    sim_scores_A_to_B = torch.empty(Ta)
    sim_scores_B_to_A = torch.empty(Tb)

    for k in range(Ta):
        max_sim_score = -float("inf")
        sim_score = (hA[k] @ hB.T) / temperature
        for tb in range(Tb):
            if (sim_score[tb] > max_sim_score).item() and maskB[tb]:
                max_sim_score = sim_score[tb]
        sim_scores_A_to_B[k] = max_sim_score

    for k in range(Tb):
        max_sim_score = -float("inf")
        sim_score = hB[k] @ hA.T / temperature
        for ta in range(Ta):
            if (sim_score[ta] > max_sim_score).item() and maskA[ta]:
                max_sim_score = sim_score[ta]
        sim_scores_B_to_A[k] = max_sim_score

    sim_score_A = torch.sum(sim_scores_A_to_B * maskA) / torch.sum(maskA)
    sim_score_B = torch.sum(sim_scores_B_to_A * maskB) / torch.sum(maskB)

    ssA_main, ssB_main = filip_similarity_score(
        hA[None, None],
        hB[None, None],
        maskA[None, None],
        maskB[None, None],
        temperature,
    )
    assert torch.isclose(ssA_main, sim_score_A.squeeze())
    assert torch.isclose(ssB_main, sim_score_B.squeeze())


def test_similarity_score_multi_batch():
    """Compare similarity scoring from filip to loopy version.
    It may actually make sense to collapse the group and the batch into a single dimension,
    but I don't think it makes a huge efficiency difference and I want to avoid bugs from
    that mixing.
    """
    torch.manual_seed(0)
    m, n = 2, 3  # number of groups
    bA, bB = 4, 8  # batch size
    Ta, Tb = 32, 32  # sequence steps
    d = 16  # embedding dimension

    hA = torch.randn(m, bA, Ta, d)
    hB = torch.randn(n, bB, Tb, d)
    maskA = torch.rand(m, bA, Ta) > 0.2
    maskB = torch.rand(n, bB, Tb) > 0.2
    temperature = torch.randn(1).item()

    sim_scores_A_to_B = torch.empty(m, n, bA, bB, Ta)
    sim_scores_B_to_A = torch.empty(m, n, bA, bB, Tb)

    for groupA, groupB in product(range(m), range(n)):
        for batchA, batchB in product(range(bA), range(bB)):
            for k in range(Ta):
                max_sim_score = -float("inf")
                sim_score = (hA[groupA, batchA, k] @ hB[groupB, batchB].T) / temperature
                for tb in range(Tb):
                    if (sim_score[tb] > max_sim_score).item() and maskB[
                        groupB, batchB, tb
                    ]:
                        max_sim_score = sim_score[tb]
                sim_scores_A_to_B[groupA, groupB, batchA, batchB, k] = max_sim_score

    for groupA, groupB in product(range(m), range(n)):
        for batchA, batchB in product(range(bA), range(bB)):
            for k in range(Tb):
                max_sim_score = -float("inf")
                sim_score = hB[groupB, batchB, k] @ hA[groupA, batchA].T / temperature
                for ta in range(Ta):
                    if (sim_score[ta] > max_sim_score).item() and maskA[
                        groupA, batchA, ta
                    ]:
                        max_sim_score = sim_score[ta]
                sim_scores_B_to_A[groupA, groupB, batchA, batchB, k] = max_sim_score

    sim_score_A = (
        torch.sum(sim_scores_A_to_B * maskA[:, None, :, None, :], dim=-1)
        / torch.sum(maskA, dim=-1)[:, None, :, None]
    )
    sim_score_B = (
        torch.sum(sim_scores_B_to_A * maskB[None, :, None, :, :], dim=-1)
        / torch.sum(maskB, dim=-1)[None, :, None, :]
    )

    ssA_main, ssB_main = filip_similarity_score(hA, hB, maskA, maskB, temperature)

    assert torch.allclose(sim_score_A, ssA_main)
    assert torch.allclose(sim_score_B, ssB_main)


if __name__ == "__main__":
    # test_decoupled_contrastive_loss()
    # test_token_dropout()

    # test_similarity_score_single_batch()
    test_similarity_score_multi_batch()
