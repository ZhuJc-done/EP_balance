"""The sync-free grouped expert MLP must match a per-expert loop in both output and gradient."""

import torch

from eplb.integration.grouped_mlp import grouped_expert_mlp, make_batched_gated_mlp


def _batched_relu_mlp(x, w):
    # x [S, N, H]; w0 [S, H, F]; w1 [S, F, H]  (plain x@w convention, matches the loop below)
    return torch.bmm(torch.relu(torch.bmm(x, w[0])), w[1])


def _loop_reference(recv_tokens, recv_slot, W0, W1, S):
    out = torch.zeros(recv_tokens.shape[0], W1.shape[-1])
    for s in range(S):
        midx = torch.nonzero(recv_slot == s, as_tuple=False).flatten()
        if midx.numel() == 0:
            continue
        out = out.index_copy(0, midx, torch.relu(recv_tokens[midx] @ W0[s]) @ W1[s])
    return out


def test_grouped_matches_loop_output_and_grad():
    torch.manual_seed(0)
    S, H, F, T = 5, 8, 16, 200
    recv_tokens = torch.randn(T, H, requires_grad=True)
    recv_tokens_ref = recv_tokens.detach().clone().requires_grad_(True)
    recv_slot = torch.randint(0, S, (T,), dtype=torch.int64)
    group_sizes = torch.bincount(recv_slot, minlength=S).to(torch.int64)
    cap = int(group_sizes.max().item()) + 4

    W0 = torch.randn(S, H, F) * 0.1
    W1 = torch.randn(S, F, H) * 0.1
    W0g = W0.clone().requires_grad_(True)
    W1g = W1.clone().requires_grad_(True)
    W0r = W0.clone().requires_grad_(True)
    W1r = W1.clone().requires_grad_(True)

    got = grouped_expert_mlp(
        recv_tokens, recv_slot, group_sizes, (W0g, W1g), _batched_relu_mlp, cap
    )
    ref = _loop_reference(recv_tokens_ref, recv_slot, W0r, W1r, S)
    assert torch.allclose(got, ref, atol=1e-5), f"output max diff {float((got-ref).abs().max())}"

    got.sum().backward()
    ref.sum().backward()
    assert torch.allclose(recv_tokens.grad, recv_tokens_ref.grad, atol=1e-5)
    assert torch.allclose(W0g.grad, W0r.grad, atol=1e-5)
    assert torch.allclose(W1g.grad, W1r.grad, atol=1e-5)


def test_batched_gated_mlp_matches_per_expert():
    import torch.nn.functional as Fnn

    torch.manual_seed(1)
    S, H, F, N = 3, 8, 16, 7
    fn = make_batched_gated_mlp(gated=True, act=Fnn.silu)
    x = torch.randn(S, N, H)
    W1 = torch.randn(S, 2 * F, H)  # Megatron [out, in]
    W2 = torch.randn(S, H, F)
    got = fn(x, (W1, W2))

    for s in range(S):
        h = x[s] @ W1[s].t()
        gate, up = torch.chunk(h, 2, dim=-1)
        ref = (Fnn.silu(gate) * up) @ W2[s].t()
        assert torch.allclose(got[s], ref, atol=1e-5)


def test_empty_slots_ok():
    torch.manual_seed(2)
    S, H, F, T = 4, 6, 10, 30
    recv_tokens = torch.randn(T, H)
    recv_slot = torch.zeros(T, dtype=torch.int64)  # all tokens in slot 0; slots 1..3 empty
    group_sizes = torch.bincount(recv_slot, minlength=S).to(torch.int64)
    cap = T + 2
    W0 = torch.randn(S, H, F) * 0.1
    W1 = torch.randn(S, F, H) * 0.1
    got = grouped_expert_mlp(recv_tokens, recv_slot, group_sizes, (W0, W1), _batched_relu_mlp, cap)
    ref = _loop_reference(recv_tokens, recv_slot, W0, W1, S)
    assert torch.allclose(got, ref, atol=1e-5)
