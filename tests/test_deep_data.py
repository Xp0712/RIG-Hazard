import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rig_hazard.deep_data import (
    DeepCacheBatchSource,
    DeepHazardWindowDataset,
    batched_multistep_targets,
    contiguous_history_mask,
    multistep_targets,
    uniform_negative_subsample_positions,
    validate_feature_contract,
)


class DeepTargetTests(unittest.TestCase):
    def test_uniform_negative_sampling_does_not_enrich_hard_negatives(self) -> None:
        strata = np.asarray([0] * 10 + [1] * 30 + [2] * 60, dtype=np.int8)
        positions, weights, summary = uniform_negative_subsample_positions(strata, 40, seed=7)
        selected = strata[positions]

        self.assertEqual(int((selected == 0).sum()), 10)
        self.assertEqual(int((selected != 0).sum()), 30)
        self.assertLess(int((selected == 1).sum()), int((selected == 2).sum()))
        self.assertAlmostEqual(float(weights.sum()), 100.0, places=4)
        self.assertEqual(summary["strategy"], "uniform_negative")

    def test_contiguous_history_restarts_after_gap(self) -> None:
        minute = 60 * 1_000_000_000
        times = np.asarray([0, 10, 20, 40, 50, 60], dtype=np.int64) * minute
        result = contiguous_history_mask(times, history_steps=3, step_minutes=10)
        np.testing.assert_array_equal(result, [False, False, True, False, False, True])

    def test_multistep_target_stops_after_first_event(self) -> None:
        minute = 60 * 1_000_000_000
        times = np.arange(7, dtype=np.int64) * 10 * minute
        labels, mask, event_step = multistep_targets(
            np.ones(7, dtype=np.int8),
            np.asarray([0, 0, 1, 0, 0, 0, 0], dtype=np.int8),
            times,
            start_index=0,
            horizon_steps=6,
            step_minutes=10,
        )
        np.testing.assert_array_equal(labels, [0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(mask, [1, 1, 1, 0, 0, 0])
        self.assertEqual(event_step, 2)

    def test_multistep_target_respects_right_censoring(self) -> None:
        minute = 60 * 1_000_000_000
        times = np.arange(6, dtype=np.int64) * 10 * minute
        labels, mask, event_step = multistep_targets(
            np.asarray([1, 1, 1, 0, 1, 1], dtype=np.int8),
            np.zeros(6, dtype=np.int8),
            times,
            start_index=0,
            horizon_steps=6,
            step_minutes=10,
        )
        np.testing.assert_array_equal(labels, np.zeros(6))
        np.testing.assert_array_equal(mask, [1, 1, 1, 0, 0, 0])
        self.assertEqual(event_step, -1)

    def test_vectorized_targets_match_single_sample_contract(self) -> None:
        minute = 60 * 1_000_000_000
        times = np.arange(9, dtype=np.int64) * 10 * minute
        risk = np.asarray([1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int8)
        hazard = np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
        starts = np.asarray([0, 2, 4], dtype=np.int64)
        labels, masks, steps = batched_multistep_targets(risk, hazard, times, starts, 5, 10)
        for index, start in enumerate(starts):
            expected = multistep_targets(risk, hazard, times, int(start), 5, 10)
            np.testing.assert_array_equal(labels[index], expected[0])
            np.testing.assert_array_equal(masks[index], expected[1])
            self.assertEqual(int(steps[index]), expected[2])

    def test_feature_contract_rejects_label_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feature_contract.json").write_text(
                json.dumps({"forbidden_as_model_inputs": ["hazard_label"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_feature_contract(root, ["temperature_mean", "hazard_label"])


class DeepDatasetTests(unittest.TestCase):
    def test_batch_source_filters_files_and_applies_seen_station_affine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features").mkdir()
            (root / "targets").mkdir()
            minute = 60 * 1_000_000_000
            manifest_files = []
            for file_id, offset in ((0, 0.0), (1, 10.0)):
                features = np.column_stack(
                    [np.arange(6, dtype=np.float32) + offset, np.ones(6, dtype=np.float32)]
                )
                feature_path = f"features/{file_id}.npy"
                target_path = f"targets/{file_id}.npz"
                np.save(root / feature_path, features, allow_pickle=False)
                targets = {
                    "issue_time_ns": np.arange(6, dtype=np.int64) * 10 * minute,
                    "risk_set": np.ones(6, dtype=np.int32),
                    "hazard_label": np.zeros(6, dtype=np.int32),
                    "station_index": np.full(6, file_id, dtype=np.int32),
                    "city_index": np.full(6, file_id, dtype=np.int32),
                    "risk_spell_index": np.ones(6, dtype=np.int32),
                }
                for name in [
                    "hard_negative_1h",
                    "hard_negative_3h",
                    "hard_negative_6h",
                    "exposure_e1_cold_humid",
                    "exposure_e2_fog_low_visibility",
                    "exposure_e3_any",
                ]:
                    targets[name] = np.zeros(6, dtype=np.int32)
                np.savez_compressed(root / target_path, **targets)
                manifest_files.append(
                    {
                        "file_id": file_id,
                        "feature_path": feature_path,
                        "target_path": target_path,
                        "station_code": str(file_id),
                    }
                )
            (root / "sample_contract.json").write_text(
                json.dumps(
                    {
                        "history_steps": 2,
                        "horizon_steps": 2,
                        "step_minutes": 10,
                        "feature_count": 2,
                        "feature_names": ["continuous", "binary"],
                    }
                ),
                encoding="utf-8",
            )
            (root / "timeline_manifest.json").write_text(
                json.dumps({"files": manifest_files}), encoding="utf-8"
            )
            np.savez_compressed(
                root / "index_train.npz",
                file_id=np.asarray([0, 1], dtype=np.int16),
                row_index=np.asarray([2, 2], dtype=np.int32),
                sample_weight=np.ones(2, dtype=np.float32),
                stratum=np.zeros(2, dtype=np.int8),
            )

            source = DeepCacheBatchSource(
                root,
                "train",
                include_file_ids={1},
                feature_center=np.asarray([10.0, 0.0], dtype=np.float32),
                feature_scale=np.asarray([2.0, 1.0], dtype=np.float32),
                prefetch_batches=2,
            )
            batch = next(source.iter_batches(batch_size=4, shuffle=False))
            self.assertEqual(len(source), 1)
            self.assertEqual(batch["file_id"].tolist(), [1])
            np.testing.assert_allclose(
                batch["history"].numpy()[0],
                np.asarray([[0.5, 1.0], [1.0, 1.0]], dtype=np.float32),
            )

    def test_history_ends_at_issue_row_and_future_is_target_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "features" / "2022").mkdir(parents=True)
            (root / "targets" / "2022").mkdir(parents=True)
            features = np.arange(16, dtype=np.float32).reshape(8, 2)
            np.save(root / "features" / "2022" / "A.npy", features, allow_pickle=False)
            minute = 60 * 1_000_000_000
            targets = {
                "issue_time_ns": np.arange(8, dtype=np.int64) * 10 * minute,
                "risk_set": np.ones(8, dtype=np.int32),
                "hazard_label": np.asarray([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int32),
                "next_recurrent_event_index": np.full(8, 7, dtype=np.int32),
                "station_index": np.full(8, 2, dtype=np.int32),
                "city_index": np.full(8, 3, dtype=np.int32),
                "risk_spell_index": np.full(8, 4, dtype=np.int32),
            }
            for name in [
                "hard_negative_1h",
                "hard_negative_3h",
                "hard_negative_6h",
                "exposure_e1_cold_humid",
                "exposure_e2_fog_low_visibility",
                "exposure_e3_any",
            ]:
                targets[name] = np.zeros(8, dtype=np.int32)
            np.savez_compressed(root / "targets" / "2022" / "A.npz", **targets)
            (root / "sample_contract.json").write_text(
                json.dumps({"history_steps": 3, "horizon_steps": 4, "step_minutes": 10}), encoding="utf-8"
            )
            (root / "timeline_manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "file_id": 0,
                                "feature_path": "features/2022/A.npy",
                                "target_path": "targets/2022/A.npz",
                                "station_code": "A",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                root / "index_train.npz",
                file_id=np.asarray([0], dtype=np.int16),
                row_index=np.asarray([3], dtype=np.int32),
                sample_weight=np.asarray([1.0], dtype=np.float32),
                stratum=np.asarray([0], dtype=np.int8),
            )

            sample = DeepHazardWindowDataset(root, "train")[0]
            np.testing.assert_array_equal(sample["history"].numpy(), features[1:4])
            np.testing.assert_array_equal(sample["hazard_target"].numpy(), [0, 0, 1, 0])
            np.testing.assert_array_equal(sample["risk_mask"].numpy(), [1, 1, 1, 0])
            self.assertEqual(int(sample["future_event_step"]), 2)
            self.assertEqual(sample["target_event_id"], "A-E0007")


if __name__ == "__main__":
    unittest.main()
