import torch

from eplb.routing_stats import compute_routing_stats


def test_routing_stats_separate_local_intra_and_cross_domain_rerouting():
    # Two 2-rank domains. Expert 0 is homed on rank 0 and expert 1 on rank 2.
    omega = torch.tensor(
        [
            [5, 10],
            [0, 0],
            [4, 0],
            [0, 0],
        ],
        dtype=torch.int64,
    )
    q = torch.zeros((4, 2, 4), dtype=torch.int64)
    q[0, 0, 2] = 5   # introduce cross-domain traffic
    q[0, 1, 1] = 10  # replace cross-domain home traffic with intra-domain traffic
    q[2, 0, 2] = 4   # replace cross-domain home traffic with local execution
    main_rank = torch.tensor([0, 2], dtype=torch.int64)
    domains = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    cost = torch.tensor(
        [
            [0, 1, 8, 8],
            [1, 0, 8, 8],
            [8, 8, 0, 1],
            [8, 8, 1, 0],
        ],
        dtype=torch.int64,
    )

    stats = compute_routing_stats(
        omega, q, main_rank, domains, s_tok=1024, cost=cost
    )

    assert stats["total_assignments"] == 19
    assert stats["baseline_local_assignments"] == 5
    assert stats["baseline_intra_domain_assignments"] == 0
    assert stats["baseline_inter_domain_assignments"] == 14
    assert stats["planned_local_assignments"] == 4
    assert stats["planned_intra_domain_assignments"] == 10
    assert stats["planned_inter_domain_assignments"] == 5
    assert stats["rerouted_assignments"] == 19
    assert stats["rerouted_local_assignments"] == 4
    assert stats["rerouted_intra_domain_assignments"] == 10
    assert stats["rerouted_inter_domain_assignments"] == 5
    assert stats["avoided_inter_domain_assignments"] == 14
    assert stats["remaining_inter_domain_assignments"] == 0
    assert stats["introduced_inter_domain_assignments"] == 5
    assert stats["baseline_topology_cost"] == 112
    assert stats["planned_topology_cost"] == 50
    assert stats["planned_one_way_inter_domain_bytes"] == 5 * 1024
    assert stats["planned_training_inter_domain_bytes"] == 4 * 5 * 1024
    assert stats["baseline_rank_flow"].tolist() == [
        [5, 0, 10, 0],
        [0, 0, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert stats["planned_rank_flow"].tolist() == [
        [0, 10, 5, 0],
        [0, 0, 0, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 0],
    ]


def test_routing_stats_reject_non_conserving_quota():
    omega = torch.tensor([[2]], dtype=torch.int64)
    q = torch.tensor([[[1]]], dtype=torch.int64)

    try:
        compute_routing_stats(
            omega,
            q,
            torch.tensor([0]),
            torch.tensor([0]),
            s_tok=16,
        )
    except ValueError as error:
        assert "does not conserve" in str(error)
    else:
        raise AssertionError("non-conserving q was accepted")
