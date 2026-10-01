import numpy as np
import pytest

import benchmark_bbrl as bench


def test_both_branches_complete_before_selection_and_keep_distinct_graphs(monkeypatch):
    a = np.zeros((4, 4), dtype=np.uint8)
    b = a.copy()
    a[0, 1] = a[1, 0] = 1
    b[0, 2] = b[2, 0] = 1
    monkeypatch.setattr(bench, "build_bundle", lambda *args: (0, [
        ("cayley", a, 2., {}, 0), ("envelope", b, 1., {}, 0)]))

    class Runner:
        def __init__(self):
            self.calls = []
            self.work = np.zeros_like(a)

        def complete(self, adj, m, branch):
            self.calls.append(branch["family"])
            self.work[:] = adj
            return self.work, 3. if branch["family"] == "cayley" else 4.

    runner = Runner()
    row, branches = bench.run_bbrl(runner, bench.DesignConfig(), 4, 4, 0)
    assert runner.calls == ["cayley", "envelope"]
    assert row["selected_family"] == "envelope"
    assert row["lambda2"] == 4.
    assert np.array_equal(branches[0]["final"], a)
    assert row["runtime_total_s"] == pytest.approx(sum(row[k] for k in (
        "runtime_init_s", "runtime_complete_s", "runtime_select_s")))


def test_mac_keeps_recorded_parameters_and_seed(monkeypatch):
    calls = []

    def complete(adj, m, **kwargs):
        calls.append(kwargs)
        return adj

    monkeypatch.setattr(bench, "mac_complete", complete)
    row, _ = bench.run_mac(bench.MAC_PARAMETERS, 4, 4, 2)
    assert calls == [dict(fw_iters=50, duality_gap_tol=1e-6, init_mode="fiedler_topk",
                          seed=53633*4 + 4 + 9973*2)]
    assert row["runtime_total_s"] >= row["runtime_complete_s"] > 0


def test_grid_preserves_paper_counts_and_excludes_endpoints():
    assert [len(bench.targets(n, 100)) for n in (8, 16, 32, 64, 128)] == [20, 100, 100, 100, 100]
    assert bench.targets(8, 100) == list(range(8, 28))
    assert bench.targets(64, 100)[-1] == 2015


def test_independent_verification_rejects_wrong_score():
    initial = bench.path_graph(4)
    row = dict(n=4, m=3, runtime_total_s=1., runtime_init_s=.5,
               runtime_complete_s=.5, runtime_select_s=0.)
    branch = dict(family="path", initial=initial, final=initial, lambda2=3.)
    with pytest.raises(AssertionError, match="Incorrect terminal score"):
        bench.verify(row, [branch])
