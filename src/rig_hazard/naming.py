"""Canonical names and frozen-artifact compatibility helpers.

Only this module knows identifiers emitted by experiments before the naming
cleanup.  Runtime code and newly written artifacts use semantic names.
"""

from __future__ import annotations

from typing import Mapping, TypeVar


ArtifactValue = TypeVar("ArtifactValue")


PROJECT_NAME = "RIG-Hazard"
MAIN_MODEL_NAME = "local_weather_hazard"
MAIN_MODEL_TRAJECTORY_DIRECTORY = "local_weather_hazard_trajectory"


# The tuple values are read-only aliases found in frozen checkpoints and tables.
# Keep them at the I/O boundary; never use them for newly written artifacts.
FROZEN_ARTIFACT_ALIASES: dict[str, tuple[str, ...]] = {
    "cold_humid_rule": ("m0_cold_humid",),
    "strict_condensation_rule": ("m0_strict_condensation",),
    MAIN_MODEL_NAME: ("m1_cloglog_hazard", "m1_hazard", "rec_none"),
    "hierarchical_barrier": ("m2_barrier",),
    "sparse_lag_graph": ("m3_graph",),
    "sparse_lag_graph_ablation": ("m3_graph_ablation",),
    "unfiltered_graph": ("m3_original",),
    "unfiltered_graph_ablation": ("m3_original_ablation",),
    "stable_graph": ("m4_stable",),
    "stable_graph_ablation": ("m4_stable_ablation",),
    "weather_only": ("m0_weather",),
    "recurrence_state": ("m1_state",),
    "weather_recurrence_interaction": ("m2_interaction",),
    "event_history": ("m3_history",),
    "gru_weather_reference": ("e1_gru_full",),
    "fast_weather_encoder": ("e1_fast_only",),
    "slow_weather_encoder": ("e1_slow_only",),
    "dual_weather_encoder": ("e1_dual_weather",),
    "dual_weather_recurrence": ("e1_dual_rec_full",),
    "gated_dual_weather_recurrence": ("e1_dual_rec_gate",),
    "gru_hierarchical_barrier": ("deep_rig_hazard_barrier",),
}

# Selected recurrence experiments use the non-history ablation as the frozen
# source for the canonical main model. This constant is for artifact lookup only.
FROZEN_MAIN_MODEL_ARTIFACT_ID = FROZEN_ARTIFACT_ALIASES[MAIN_MODEL_NAME][2]

FROZEN_RESULT_COMPONENT_ALIASES: dict[str, str] = {
    "deep_rig_hazard": "deep_model_experiments",
    "e0_protocol_audit": "protocol_audit",
    "e1_dual_weather": "weather_ablation",
    "e1_e3_bootstrap_2022": "selection_year_bootstrap",
    "e1_e3_bootstrap_locked": "locked_year_bootstrap",
    "e3_recurrence_gate": "recurrence_gate_ablation",
    "e4_nested_alert": "nested_alert",
    "e6_selected_full_trajectory": "selected_model_trajectory",
    "final": "trained_models",
    "local_hazard_baselines": "local_weather_hazard_baselines",
    "paper_figures": "manuscript_figures",
    "preprocessed_10min": "preprocessed_data_10min",
    "recurrence_experiments": "recurrence_analysis",
    "recurrence_next_stage": "recurrence_modeling",
    "recurrent_icing_spec": "icing_model_experiments",
    "remaining_experiments": "supplementary_experiments",
    "seasonal_risk_structure_check": "seasonal_risk_structure_validation",
}


def canonicalize_artifact_name(name: str) -> str:
    """Translate a frozen model/column identifier to its semantic name."""

    for canonical_name, aliases in FROZEN_ARTIFACT_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if name == alias:
                return canonical_name
            for separator in ("_", "::"):
                prefix = f"{alias}{separator}"
                if name.startswith(prefix):
                    return f"{canonical_name}{separator}{name[len(prefix):]}"
    return name


def artifact_name_candidates(canonical_name: str) -> tuple[str, ...]:
    """Return the current identifier followed by any frozen aliases."""

    return (canonical_name, *FROZEN_ARTIFACT_ALIASES.get(canonical_name, ()))


def canonicalize_project_relative_path(value: str) -> str:
    """Translate a frozen project-relative result path to the current layout."""

    normalized = value.replace("\\", "/")
    visualization_prefixes = (
        "rig_hazard_outputs/paper_figures",
        "results/paper_figures",
        "results/manuscript_figures",
    )
    visualization_prefix = next(
        (prefix for prefix in visualization_prefixes if normalized.startswith(prefix)),
        None,
    )
    if visualization_prefix is not None:
        normalized = normalized.replace(
            visualization_prefix, "visualization/manuscript_figures", 1
        )
    elif normalized == "rig_hazard_outputs" or normalized.startswith(
        "rig_hazard_outputs/"
    ):
        normalized = normalized.replace("rig_hazard_outputs", "results", 1)
    if not normalized.startswith(("results/", "visualization/")):
        return normalized
    parts = [
        FROZEN_RESULT_COMPONENT_ALIASES.get(part, part)
        for part in normalized.split("/")
    ]
    return "/".join(parts)


def canonical_column_renames(columns: list[str]) -> dict[str, str]:
    """Return only the legacy-to-canonical column renames that are required."""

    return {
        column: canonical_name
        for column in columns
        if (canonical_name := canonicalize_artifact_name(column)) != column
    }


def resolve_artifact_name(available_names: list[str], canonical_name: str) -> str:
    """Resolve a canonical field name against a frozen artifact schema."""

    if canonical_name in available_names:
        return canonical_name
    for available_name in available_names:
        if canonicalize_artifact_name(available_name) == canonical_name:
            return available_name
    raise KeyError(canonical_name)


def artifact_value(
    mapping: Mapping[str, ArtifactValue], canonical_name: str
) -> ArtifactValue:
    """Read a canonical mapping key with a fallback for frozen artifacts."""

    if canonical_name in mapping:
        return mapping[canonical_name]
    for alias in FROZEN_ARTIFACT_ALIASES.get(canonical_name, ()):
        if alias in mapping:
            return mapping[alias]
    raise KeyError(canonical_name)
