"""Tests for semantic names and frozen-artifact compatibility."""

from __future__ import annotations

import unittest

from rig_hazard.naming import (
    FROZEN_ARTIFACT_ALIASES,
    MAIN_MODEL_NAME,
    artifact_name_candidates,
    artifact_value,
    canonical_column_renames,
    canonicalize_artifact_name,
    canonicalize_project_relative_path,
)


class NamingTests(unittest.TestCase):
    def test_frozen_main_model_alias_maps_to_canonical_name(self) -> None:
        alias = FROZEN_ARTIFACT_ALIASES[MAIN_MODEL_NAME][1]
        self.assertEqual(canonicalize_artifact_name(alias), MAIN_MODEL_NAME)
        self.assertEqual(
            canonicalize_artifact_name(f"{alias}_6h"),
            f"{MAIN_MODEL_NAME}_6h",
        )

    def test_artifact_mapping_prefers_canonical_key(self) -> None:
        alias = FROZEN_ARTIFACT_ALIASES[MAIN_MODEL_NAME][0]
        self.assertEqual(
            artifact_value({alias: "frozen", MAIN_MODEL_NAME: "current"}, MAIN_MODEL_NAME),
            "current",
        )
        self.assertEqual(artifact_value({alias: "frozen"}, MAIN_MODEL_NAME), "frozen")

    def test_column_renames_and_candidates_cover_frozen_schema(self) -> None:
        alias = FROZEN_ARTIFACT_ALIASES["stable_graph"][0]
        legacy_column = f"{alias}_eta"
        self.assertEqual(
            canonical_column_renames([legacy_column]),
            {legacy_column: "stable_graph_eta"},
        )
        self.assertEqual(artifact_name_candidates("stable_graph")[0], "stable_graph")

    def test_frozen_result_path_maps_to_current_layout(self) -> None:
        weather_alias = FROZEN_ARTIFACT_ALIASES["dual_weather_encoder"][0]
        frozen_path = f"rig_hazard_outputs/recurrent_icing_spec/{weather_alias}"
        self.assertEqual(
            canonicalize_project_relative_path(frozen_path),
            "results/icing_model_experiments/weather_ablation",
        )


if __name__ == "__main__":
    unittest.main()
