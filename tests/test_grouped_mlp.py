"""The sync-free grouped expert MLP must match a per-expert loop in both output and gradient."""

import pytest
import torch

from eplb.integration.grouped_mlp import (
    grouped_expert_mlp,
    grouped_mm_usable,
    make_batched_gated_mlp,
    make_grouped_gated_mlp,
)

requires_grouped_mm = pytest.mark.skipif(
    not (torch.cuda.is_available() and grouped_mm_usable(torch.empty(1, device="cuda", dtype=torch.bfloat16))),
    reason="ragged path needs torch._grouped_mm (CUDA SM90+, 16-bit float)",
)


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


def test_elastic_padding_is_excluded_from_output_and_grad():
    """Worst-case rows never enter expert forward, Dgrad or Wgrad."""
    torch.manual_seed(3)
    S, H, F, actual, padded = 4, 6, 10, 17, 32
    slot = torch.randint(0, S, (actual,), dtype=torch.int64)
    recv_slot = torch.cat([slot, torch.zeros(padded - actual, dtype=torch.int64)])
    valid = torch.arange(padded) < actual
    sizes = torch.bincount(slot, minlength=S).to(torch.int64)
    cap = actual

    x = torch.randn(padded, H, requires_grad=True)
    xr = x[:actual].detach().clone().requires_grad_(True)
    w0 = torch.randn(S, H, F) * 0.1
    w1 = torch.randn(S, F, H) * 0.1
    w0g, w1g = w0.clone().requires_grad_(True), w1.clone().requires_grad_(True)
    w0r, w1r = w0.clone().requires_grad_(True), w1.clone().requires_grad_(True)

    got = grouped_expert_mlp(
        x, recv_slot, sizes, (w0g, w1g), _batched_relu_mlp, cap, valid_mask=valid
    )
    ref = _loop_reference(xr, slot, w0r, w1r, S)
    assert torch.allclose(got[:actual], ref, atol=1e-5)
    assert torch.count_nonzero(got[actual:]) == 0

    got.sum().backward()
    ref.sum().backward()
    assert torch.allclose(x.grad[:actual], xr.grad, atol=1e-5)
    assert torch.count_nonzero(x.grad[actual:]) == 0
    assert torch.allclose(w0g.grad, w0r.grad, atol=1e-5)
    assert torch.allclose(w1g.grad, w1r.grad, atol=1e-5)


def _ragged_case(S, H, F, actual, padded, seed):
    """Shared fixture: a padded-vs-ragged comparison on identical bf16 inputs."""
    torch.manual_seed(seed)
    dev = "cuda"
    slot = torch.randint(0, S, (actual,), dtype=torch.int64, device=dev)
    slot, _ = slot.sort()  # any order is legal; sorted keeps the loop reference simple
    recv_slot = torch.cat([slot, torch.zeros(padded - actual, dtype=torch.int64, device=dev)])
    valid = torch.arange(padded, device=dev) < actual
    sizes = torch.bincount(slot, minlength=S).to(torch.int64)

    x = torch.randn(padded, H, device=dev, dtype=torch.bfloat16)
    w1 = (torch.randn(S, 2 * F, H, device=dev, dtype=torch.bfloat16) * 0.05)
    w2 = (torch.randn(S, H, F, device=dev, dtype=torch.bfloat16) * 0.05)
    return recv_slot, sizes, valid, x, w1, w2


@requires_grouped_mm
@pytest.mark.parametrize("padded,actual", [(512, 512), (512, 301)])
def test_ragged_matches_padded_forward_and_grad(padded, actual):
    """The ragged grouped GEMM must reproduce the padded bmm it replaces, gradients included."""
    S, H, F = 8, 128, 256
    recv_slot, sizes, valid, x, w1, w2 = _ragged_case(S, H, F, actual, padded, seed=7)
    cap = max(int(sizes.max()), 1)
    act = torch.nn.functional.silu

    def run(use_ragged):
        xg = x.detach().clone().requires_grad_(True)
        w1g = w1.detach().clone().requires_grad_(True)
        w2g = w2.detach().clone().requires_grad_(True)
        out = grouped_expert_mlp(
            xg, recv_slot, sizes, (w1g, w2g),
            make_batched_gated_mlp(True, act), cap, valid_mask=valid,
            grouped_mlp_fn=make_grouped_gated_mlp(True, act) if use_ragged else None,
        )
        out.float().pow(2).sum().backward()
        return out, xg.grad, w1g.grad, w2g.grad

    ragged = run(True)
    padded_out = run(False)
    for name, a, b in zip(("out", "grad_x", "grad_w1", "grad_w2"), ragged, padded_out):
        scale = max(b.float().abs().max().item(), 1e-6)
        diff = (a.float() - b.float()).abs().max().item()
        assert diff / scale < 2e-2, f"{name} rel diff {diff / scale:.3e}"

    # ElasticBuffer's worst-case rows must stay out of both the output and Dgrad
    assert torch.count_nonzero(ragged[0][actual:]) == 0
    assert torch.count_nonzero(ragged[1][actual:]) == 0


@requires_grouped_mm
def test_ragged_path_does_not_sync_host():
    """The ragged path must launch without reading any device value to host."""
    S, H, F, actual, padded = 8, 128, 256, 301, 512
    recv_slot, sizes, valid, x, w1, w2 = _ragged_case(S, H, F, actual, padded, seed=11)
    act = torch.nn.functional.silu
    fn = make_grouped_gated_mlp(True, act)
    xg = x.detach().clone().requires_grad_(True)
    w1g = w1.detach().clone().requires_grad_(True)
    w2g = w2.detach().clone().requires_grad_(True)

    def step():
        out = grouped_expert_mlp(
            xg, recv_slot, sizes, (w1g, w2g),
            make_batched_gated_mlp(True, act), None, valid_mask=valid, grouped_mlp_fn=fn,
        )
        out.float().pow(2).sum().backward()

    step()  # warm up autograd/cuBLAS outside the strict window
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@requires_grouped_mm
@pytest.mark.parametrize("swap", [False, True])
def test_ragged_mm_tn_matches_batched_wgrad(swap):
    """The weight-gradient product must match the padded bmm in both operand orders.

    ``transpose_w`` (Megatron ``[out, in]``) swaps these operands so the Wgrad lands already in
    parameter layout; getting the swap wrong silently transposes every expert's gradient.
    """
    from eplb.integration.grouped_mlp import ragged_mm_tn

    torch.manual_seed(17)
    S, M, N, cap = 6, 128, 256, 64
    sizes = torch.tensor([64, 0, 31, 64, 7, 40], device="cuda", dtype=torch.int64)
    T = int(sizes.sum())
    a_pad = torch.zeros(S, cap, M, device="cuda", dtype=torch.bfloat16)
    b_pad = torch.zeros(S, cap, N, device="cuda", dtype=torch.bfloat16)
    a_rag = torch.randn(T, M, device="cuda", dtype=torch.bfloat16)
    b_rag = torch.randn(T, N, device="cuda", dtype=torch.bfloat16)
    start = 0
    for s in range(S):
        n = int(sizes[s])
        a_pad[s, :n], b_pad[s, :n] = a_rag[start:start + n], b_rag[start:start + n]
        start += n

    offs = torch.cumsum(sizes, 0).to(torch.int32)
    if swap:
        got, ref = ragged_mm_tn(b_rag, a_rag, offs), ragged_mm_tn(b_pad, a_pad, None)
    else:
        got, ref = ragged_mm_tn(a_rag, b_rag, offs), ragged_mm_tn(a_pad, b_pad, None)

    assert got.shape == ref.shape
    scale = max(ref.float().abs().max().item(), 1e-6)
    assert (got.float() - ref.float()).abs().max().item() / scale < 2e-2


@requires_grouped_mm
def test_overlapped_experts_ragged_matches_padded(monkeypatch):
    """OverlappedExperts' hand-written backward agrees across layouts, gated + Megatron weights."""
    from eplb.integration.overlap import overlapped_grouped_expert_mlp

    torch.manual_seed(19)
    S, H, F, T = 6, 128, 256, 400
    dev = "cuda"
    slot, _ = torch.randint(0, S, (T,), dtype=torch.int64, device=dev).sort()
    sizes = torch.bincount(slot, minlength=S).to(torch.int64)
    tokens = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(S, 2 * F, H, device=dev, dtype=torch.bfloat16) * 0.05   # Megatron [out, in]
    w2 = torch.randn(S, H, F, device=dev, dtype=torch.bfloat16) * 0.05

    def run(use_ragged):
        monkeypatch.setenv("EPLB_GROUPED_GEMM", "1" if use_ragged else "0")
        x = tokens.detach().clone().requires_grad_(True)
        local = {
            e: (w1[e].clone().requires_grad_(True), w2[e].clone().requires_grad_(True))
            for e in range(S)
        }
        out = overlapped_grouped_expert_mlp(
            x, slot, sizes, local,
            slot_to_e=torch.arange(S, dtype=torch.int64, device=dev),
            main_rank=torch.zeros(S, dtype=torch.int64, device=dev),
            replicated=[], weight_shapes=[(2 * F, H), (H, F)], cap=int(sizes.max()),
            gated=True, act=torch.nn.functional.silu, transpose_w=True,
            my_rank=0, n_slot=S, group=None,
        )
        out.float().pow(2).sum().backward()
        return out, x.grad, [local[e][0].grad for e in range(S)], [local[e][1].grad for e in range(S)]

    r_out, r_gx, r_gw1, r_gw2 = run(True)
    p_out, p_gx, p_gw1, p_gw2 = run(False)

    def close(a, b, what):
        scale = max(b.float().abs().max().item(), 1e-6)
        diff = (a.float() - b.float()).abs().max().item()
        assert diff / scale < 3e-2, f"{what} rel diff {diff / scale:.3e}"

    close(r_out, p_out, "out")
    close(r_gx, p_gx, "grad_x")
    for e in range(S):
        close(r_gw1[e], p_gw1[e], f"grad_w1[{e}]")
        close(r_gw2[e], p_gw2[e], f"grad_w2[{e}]")


@requires_grouped_mm
def test_ragged_skips_empty_slots_and_padding_rows():
    """Work must track the real token count, not n_slot x cap."""
    S, H, F, actual, padded = 16, 128, 256, 200, 4096
    recv_slot, sizes, valid, x, w1, w2 = _ragged_case(S, H, F, actual, padded, seed=13)
    act = torch.nn.functional.silu
    out = grouped_expert_mlp(
        x, recv_slot, sizes, (w1, w2), make_batched_gated_mlp(True, act), None,
        valid_mask=valid, grouped_mlp_fn=make_grouped_gated_mlp(True, act),
    )
    assert out.shape == (padded, H)
    assert torch.isfinite(out[:actual]).all()      # no NaN leaking in from unwritten rows
    assert torch.count_nonzero(out[actual:]) == 0


def _budget_case(seed=23):
    """A ragged case plus the pieces every ``max_recv_rows`` test needs."""
    S, H, F, actual, padded = 8, 128, 256, 301, 512
    recv_slot, sizes, valid, x, w1, w2 = _ragged_case(S, H, F, actual, padded, seed=seed)
    return dict(
        S=S, H=H, actual=actual, padded=padded, recv_slot=recv_slot, sizes=sizes,
        valid=valid, x=x, w1=w1, w2=w2, act=torch.nn.functional.silu,
    )


@requires_grouped_mm
@pytest.mark.parametrize("budget", [512, 400, 320])
def test_max_recv_rows_matches_untrimmed(budget):
    """Trimming the row order to a budget must not change anything the experts compute.

    Every valid row already sorts ahead of the padding, so a budget at or above the real receipt
    count can only drop rows that were masked to zero anyway.
    """
    c = _budget_case()

    def run(max_recv_rows):
        xg = c["x"].detach().clone().requires_grad_(True)
        w1g = c["w1"].detach().clone().requires_grad_(True)
        w2g = c["w2"].detach().clone().requires_grad_(True)
        out = grouped_expert_mlp(
            xg, c["recv_slot"], c["sizes"], (w1g, w2g),
            make_batched_gated_mlp(True, c["act"]), None, valid_mask=c["valid"],
            grouped_mlp_fn=make_grouped_gated_mlp(True, c["act"]), max_recv_rows=max_recv_rows,
        )
        out.float().pow(2).sum().backward()
        return out, xg.grad, w1g.grad, w2g.grad

    for name, got, ref in zip(("out", "grad_x", "grad_w1", "grad_w2"), run(budget), run(None)):
        assert torch.equal(got, ref), f"{name} changed under a {budget}-row budget"


@requires_grouped_mm
def test_max_recv_rows_shrinks_the_gemm_tensors():
    """The budget, not the transport's worst case, must size what the expert GEMM sees."""
    c, budget = _budget_case(seed=29), 320
    base = make_grouped_gated_mlp(True, c["act"])
    rows = []

    def fn(x, w, offs):
        rows.append(x.shape[0])
        return base(x, w, offs)

    fn.supports = base.supports
    out = grouped_expert_mlp(
        c["x"], c["recv_slot"], c["sizes"], (c["w1"], c["w2"]),
        make_batched_gated_mlp(True, c["act"]), None, valid_mask=c["valid"],
        grouped_mlp_fn=fn, max_recv_rows=budget,
    )
    assert rows == [budget], f"expert GEMM ran on {rows} rows, expected [{budget}]"
    # the caller still gets the transport's layout back, so combine needs no change
    assert out.shape == (c["padded"], c["H"])


@requires_grouped_mm
@pytest.mark.parametrize("budget,fits", [(320, True), (301, True), (300, False), (64, False)])
def test_max_recv_rows_overflow_is_flagged_on_device(monkeypatch, budget, fits):
    """A budget below the real receipt count must trip the guard, not drop rows silently.

    The predicate is inspected rather than executed: a real ``_assert_async`` failure is a sticky
    CUDA fault that would take down every later test in this process.
    """
    c = _budget_case(seed=31)
    checked = []
    monkeypatch.setattr(torch, "_assert_async", lambda cond, *a, **k: checked.append(bool(cond)))
    grouped_expert_mlp(
        c["x"], c["recv_slot"], c["sizes"], (c["w1"], c["w2"]),
        make_batched_gated_mlp(True, c["act"]), None, valid_mask=c["valid"],
        grouped_mlp_fn=make_grouped_gated_mlp(True, c["act"]), max_recv_rows=budget,
    )
    assert checked == [fits], f"budget {budget} vs {c['actual']} received rows"


@requires_grouped_mm
def test_max_recv_rows_does_not_sync_host():
    """Enforcing the budget must stay on device -- the guard is async by design."""
    c, budget = _budget_case(seed=37), 320
    fn = make_grouped_gated_mlp(True, c["act"])
    xg = c["x"].detach().clone().requires_grad_(True)
    w1g = c["w1"].detach().clone().requires_grad_(True)
    w2g = c["w2"].detach().clone().requires_grad_(True)

    def step():
        out = grouped_expert_mlp(
            xg, c["recv_slot"], c["sizes"], (w1g, w2g),
            make_batched_gated_mlp(True, c["act"]), None, valid_mask=c["valid"],
            grouped_mlp_fn=fn, max_recv_rows=budget,
        )
        out.float().pow(2).sum().backward()

    step()  # warm up autograd/cuBLAS outside the strict window
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
