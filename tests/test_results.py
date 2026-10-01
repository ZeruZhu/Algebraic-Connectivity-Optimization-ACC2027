"""Audit the released measurements against provenance and manuscript numbers."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_results import load_results, read_csv, summarize  # noqa: E402


class ReleasedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_results()
        cls.quality, cls.cost = summarize(cls.rows)

    def test_artifact_hashes_and_current_method_sets(self):
        for name in ("paper", "baseline"):
            directory = ROOT / "results" / name
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            for filename, expected in metadata["public_artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(), expected)
            self.assertEqual(Counter(row["method"] for row in read_csv(directory / "raw.csv")), metadata["methods"])
        self.assertEqual({row["method"] for row in self.rows}, {"BB-RL", "MAC", "FVG", "MDMD", "WTS", "k-opt", "SDP-step", "OA-PMC"})

    def test_fresh_grid_matches_reference_and_has_five_repeats(self):
        grids = defaultdict(set)
        repeats = defaultdict(set)
        for row in self.rows:
            grids[row["method"]].add((row["n"], row["m"], row["repeat_idx"]))
            repeats[row["method"], row["n"], row["m"]].add(row["repeat_idx"])
        self.assertEqual(grids["BB-RL"], grids["FVG"])
        self.assertEqual(grids["MAC"], grids["FVG"])
        self.assertEqual(len(grids["BB-RL"]), 2100)
        for values in repeats.values():
            self.assertEqual(values, set(range(5)))
        for row in self.quality:
            self.assertEqual(row["density_points"], 20 if row["n"] == 8 else 100)

    def test_quality_agrees_with_published_table(self):
        # Each tuple is (delta mean, density std, win %, match %) from Table I.
        expected = {
            "BB-RL": [(0.25, .42, 40, 50), (.45, .62, 66, 11), (.73, .85, 71, 3), (1.13, 1.22, 73.2, 2), (1.61, 2, 66.6, 1.6)],
            "MAC": [(-.47, .41, 0, 25), (-1.27, .65, 1, 2), (-3.03, 1.42, 0, 2), (-6.62, 3.10, 0, 1), (-13.72, 6.52, 0, 1)],
            "MDMD": [(-.08, .35, 10, 50), (-.47, .40, 7, 12), (-1.09, .73, 4, 5), (-2.15, 1.49, 1, 3), (-3.89, 2.49, 1, 1)],
            "WTS": [(.27, .36, 50, 50), (.18, .22, 83, 12), (-.23, .31, 12, 2), (-.76, .63, 2, 2), (-1.63, 1.12, 1, 1)],
            "k-opt": [(.19, .29, 45, 55), (.20, .16, 81, 19), (.13, .13, 90, 10), (.04, .05, 80, 20), (.01, .02, 58, 8)],
            "SDP-step": [(.08, .57, 30, 40), (.29, .57, 67, 11), (.39, .71, 71, 4)],
            "OA-PMC": [(.31, .40, 55, 45)],
        }
        lookup = {(row["method"], row["n"]): row for row in self.quality}
        for method, entries in expected.items():
            for n, values in zip((8, 16, 32, 64, 128), entries):
                row = lookup[method, n]
                # Historical rows in the manuscript round percentages to integers;
                # retain their fractional values in the released CSV summaries.
                win = row["win_pct_vs_fvg"] if method in {"BB-RL", "MAC"} else round(row["win_pct_vs_fvg"])
                match = row["match_pct_vs_fvg"] if method in {"BB-RL", "MAC"} else round(row["match_pct_vs_fvg"])
                actual = (round(row["delta_lambda2_mean"], 2), round(row["delta_lambda2_std"], 2), win, match)
                self.assertEqual(actual, values, (method, n))
        for n, expected_value in zip((8, 16, 32, 64, 128), ((3.34, 1.77), (6.72, 4.18), (13.99, 8.81), (28.87, 18.19), (59.16, 36.84))):
            row = lookup["FVG", n]
            self.assertEqual((round(row["lambda2_mean"], 2), round(row["lambda2_std"], 2)), expected_value)

    def test_runtime_matches_measured_run_and_components(self):
        expected = {
            "BB-RL": [.004849613760598006, .0077593940945807835, .014974137707846238, .038396862662222704, .119731102717662],
            "MAC": [.004345736920222407, .006314014020084869, .01624501277410309, .04627781909590704, .15281702217800194],
        }
        lookup = {(row["method"], row["n"]): row for row in self.cost}
        for method, times in expected.items():
            for n, seconds in zip((8, 16, 32, 64, 128), times):
                self.assertAlmostEqual(lookup[method, n]["runtime_total_mean_s"], seconds, places=14)
        for row in self.rows:
            if row["method"] == "BB-RL":
                self.assertAlmostEqual(row["runtime_init_s"] + row["runtime_complete_s"] + row["runtime_select_s"], row["runtime_total_s"], places=12)
        for row in self.cost:
            expected_provenance = "fresh_matched" if row["method"] in {"MAC", "BB-RL"} else "historical"
            self.assertEqual(row["timing_provenance"], expected_provenance)

    def test_all_completion_branches_and_selected_results_are_accounted_for(self):
        branches = defaultdict(list)
        for row in read_csv(ROOT / "results/paper/branches.csv"):
            key = tuple(int(row[field]) for field in ("n", "m", "repeat_idx"))
            branches[key].append(row)
            self.assertLessEqual(float(row["score_error"]), 1e-10)
            self.assertLessEqual(int(row["init_edges"]), key[1])
        self.assertEqual(sum(map(len, branches.values())), 2710)
        for row in read_csv(ROOT / "results/paper/raw.csv"):
            self.assertLessEqual(float(row["max_score_error"]), 1e-10)
            if row["method"] != "BB-RL":
                continue
            key = tuple(int(row[field]) for field in ("n", "m", "repeat_idx"))
            candidates = branches[key]
            self.assertEqual(len(candidates), int(row["branches"]))
            self.assertAlmostEqual(float(row["lambda2"]), max(float(branch["lambda2"]) for branch in candidates), places=10)
            self.assertTrue(any(branch["family"] == row["selected_family"] and branch["final_sha256"] == row["graph_sha256"] for branch in candidates))

    def test_committed_summaries_reproduce_from_raw(self):
        for filename, expected in (("quality.csv", self.quality), ("cost.csv", self.cost)):
            stored = read_csv(ROOT / "results/paper" / filename)
            self.assertEqual(len(stored), len(expected))
            for actual_row, expected_row in zip(stored, expected):
                for field, value in expected_row.items():
                    if isinstance(value, float):
                        self.assertAlmostEqual(float(actual_row[field]), value, places=12)
                    else:
                        self.assertEqual(actual_row[field], str(value))


if __name__ == "__main__":
    unittest.main()
