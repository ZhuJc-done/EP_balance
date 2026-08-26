import subprocess
import sys
from pathlib import Path

import torch

from eplb.integration.plan_trace import PlanTraceWriter, rank_local_trace_path
from eplb.integration.rebalancer import EPLBRebalancer
from eplb.loads import Loads
from eplb.plan import Plan
from eplb.problem import ProblemSpec
from eplb.topology import Topology
from eval.analyze_eplb_routing import analyze


def _case():
    topology = Topology.from_nvlink_rdma(
        num_nodes=2, gpus_per_node=2, device="cpu"
    )
    spec = ProblemSpec(
        num_experts=2,
        main_rank=torch.tensor([0, 2]),
        weight_bytes=torch.tensor([1024, 1024]),
        s_tok=1024,
        n_slot=2,
    )
    omega = torch.tensor(
        [[5, 10], [0, 0], [4, 0], [0, 0]], dtype=torch.int64
    )
    q = torch.zeros((4, 2, 4), dtype=torch.int64)
    q[0, 0, 2] = 5
    q[0, 1, 1] = 10
    q[2, 0, 2] = 4
    x = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.int8
    )
    return topology, spec, Loads(omega), Plan(x=x, q=q, theta=10)


def test_plan_trace_writer_captures_applied_quota_and_analyzer_reads_it(tmp_path):
    topology, spec, loads, plan = _case()
    trace_path = tmp_path / "routing.pt"
    writer = PlanTraceWriter(
        trace_path,
        topology,
        spec,
        solver="scale",
        global_rank=0,
        pipeline_rank=0,
        max_samples=1,
        flush_every=1,
    )

    writer.append(loads, plan, layer_id=3, micro_batch_id=7)
    writer.append(loads, plan, layer_id=3, micro_batch_id=8)

    trace = torch.load(trace_path, weights_only=False)
    assert trace["meta"]["format_version"] == 4
    assert trace["meta"]["trace_kind"] == "applied_plan"
    assert trace["meta"]["plan_solver"] == "scale"
    assert len(trace["samples"]) == 1
    assert torch.equal(trace["samples"][0]["omega"], loads.omega)
    assert torch.equal(trace["samples"][0]["x"], plan.x)
    assert torch.equal(trace["samples"][0]["q"], plan.q)

    summary, samples, by_layer, rank_flow, domain_flow, quotas, placements = analyze(
        [trace_path], warmup_mb=7
    )
    assert summary["aggregate"]["planned_inter_domain_assignments"] == 5
    assert summary["aggregate"]["avoided_inter_domain_assignments"] == 14
    assert samples[0]["layer"] == 3
    assert by_layer[0]["layer"] == 3
    assert {row["route"] for row in rank_flow} == {"home", "planned"}
    assert {row["route"] for row in domain_flow} == {"home", "planned"}
    assert any(
        row["source_rank"] == 0
        and row["expert"] == 1
        and row["destination_rank"] == 1
        and row["assignments"] == 10
        for row in quotas
    )
    assert any(
        row["expert"] == 1
        and row["host_rank"] == 1
        and row["sample_fraction"] == 1.0
        for row in placements
    )

    output = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            "eval/analyze_eplb_routing.py",
            "--trace",
            str(trace_path),
            "--warmup-mb",
            "7",
            "--out-dir",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output / "summary.json").exists()
    assert (output / "samples.csv.gz").exists()
    assert (output / "rank_flow.csv").exists()
    assert (output / "domain_flow.csv").exists()
    assert (output / "quota.csv.gz").exists()
    assert (output / "placement.csv").exists()


def test_rank_local_trace_path_only_suffixes_multiple_writers(tmp_path):
    base = tmp_path / "routing.pt"
    assert rank_local_trace_path(base, global_rank=16, num_writers=1) == base
    assert rank_local_trace_path(base, global_rank=16, num_writers=2) == (
        tmp_path / "routing.rank16.pt"
    )
    assert rank_local_trace_path(
        tmp_path / "routing.{rank}.pt", global_rank=16, num_writers=2
    ) == (tmp_path / "routing.16.pt")


def test_rebalancer_observes_the_exact_plan_before_discarding_rings():
    topology, spec, loads, plan = _case()
    observed = []
    rebalancer = EPLBRebalancer(
        topology,
        spec,
        plan_solver=lambda *_: plan,
        plan_observer=lambda got_loads, got_plan, layer, mb: observed.append(
            (got_loads, got_plan, layer, mb)
        ),
        ring_size=0,
    )

    result = rebalancer.rebalance_from_omega(loads, layer_id=3, micro_batch_id=7)

    assert result.plan is plan
    assert observed == [(loads, plan, 3, 7)]
    assert rebalancer._omega_ring == {}
    assert rebalancer._plan_ring == {}
