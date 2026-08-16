from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .windows import WindowDataset


def normalize_windows(
    values: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = values.mean(axis=1)
    stds = values.std(axis=1)
    safe_stds = np.maximum(stds, epsilon)
    normalized = (values - means[:, None]) / safe_stds[:, None]
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / np.maximum(norms, epsilon)
    return normalized.astype(np.float32), means.astype(np.float32), safe_stds.astype(np.float32)


def _standardized_unit(
    candidates: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(candidates, axis=0)
    scale = np.maximum(np.nanstd(candidates, axis=0), epsilon)
    candidate_scaled = np.nan_to_num((candidates - center) / scale)
    query_scaled = np.nan_to_num((queries - center) / scale)
    candidate_norm = np.linalg.norm(candidate_scaled, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_scaled, axis=1, keepdims=True)
    candidate_unit = candidate_scaled / np.maximum(candidate_norm, epsilon)
    query_unit = query_scaled / np.maximum(query_norm, epsilon)
    return candidate_unit.astype(np.float32), query_unit.astype(np.float32)


def _eligible_minmax(values: np.ndarray, epsilon: float) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    lower = float(np.min(values[finite]))
    upper = float(np.max(values[finite]))
    if upper - lower <= epsilon:
        return np.zeros_like(values, dtype=np.float32)
    result = np.zeros_like(values, dtype=np.float32)
    result[finite] = (values[finite] - lower) / (upper - lower)
    return result


@dataclass
class KnowledgeBase:
    x: np.ndarray
    y: np.ndarray
    vectors: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    features: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]

    def save(self, arrays_path: Path, metadata_path: Path) -> None:
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            x=self.x,
            y=self.y,
            vectors=self.vectors,
            input_mean=self.input_mean,
            input_std=self.input_std,
            features=self.features,
        )
        self.metadata.to_parquet(metadata_path, index=False)
        arrays_path.with_suffix(".json").write_text(
            json.dumps({"feature_names": self.feature_names, "size": len(self.metadata)}, indent=2),
            encoding="utf-8",
        )


@dataclass
class RetrievalResult:
    prediction: np.ndarray
    weighted_prediction: np.ndarray
    spread: np.ndarray
    mean_similarity: np.ndarray
    max_similarity: np.ndarray
    candidate_count: np.ndarray
    eligible_candidate_count: np.ndarray
    selected_event_fraction: np.ndarray
    query_event_context: np.ndarray
    event_conditioning_applied: np.ndarray
    evidence: pd.DataFrame

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.prediction).all(axis=1)

    def as_features(self) -> np.ndarray:
        return np.column_stack(
            [
                self.prediction,
                self.spread,
                self.mean_similarity,
                self.max_similarity,
                self.candidate_count,
                self.selected_event_fraction,
                self.event_conditioning_applied,
            ]
        ).astype(np.float32)

    def feature_names(self, prefix: str = "retrieval") -> list[str]:
        horizon = self.prediction.shape[1]
        return (
            [f"{prefix}_trajectory_h{step:02d}" for step in range(1, horizon + 1)]
            + [f"{prefix}_spread_h{step:02d}" for step in range(1, horizon + 1)]
            + [
                f"{prefix}_mean_similarity",
                f"{prefix}_max_similarity",
                f"{prefix}_candidate_count",
                f"{prefix}_selected_event_fraction",
                f"{prefix}_event_conditioning_applied",
            ]
        )


def build_knowledge_base(
    dataset: WindowDataset,
    cfg: ProjectConfig,
    stride_hours: int | None = None,
) -> KnowledgeBase:
    stride = int(stride_hours or cfg.retrieval.kb_stride_hours)
    if stride <= 0:
        raise ValueError("Knowledge-base stride must be positive")
    train_indices = np.flatnonzero(dataset.metadata["split"].to_numpy() == "train")
    if train_indices.size == 0:
        raise ValueError("Training split is empty")
    origin_rows = dataset.metadata.loc[train_indices, "origin_row"].to_numpy(dtype=int)
    anchor = int(origin_rows.min())
    indices = train_indices[(origin_rows - anchor) % stride == 0]
    if indices.size < cfg.retrieval.k:
        raise ValueError("Knowledge base has fewer candidates than configured k")
    metadata = dataset.metadata.loc[indices].reset_index(drop=True).copy()
    metadata["kb_stride_hours"] = stride
    if not metadata["window_id"].is_unique:
        raise AssertionError("Knowledge-base window IDs must be unique")
    if not pd.to_datetime(metadata["origin_time"], utc=True).is_monotonic_increasing:
        raise AssertionError("Knowledge-base candidates must remain chronological")
    vectors, means, stds = normalize_windows(dataset.x[indices], cfg.retrieval.epsilon)
    return KnowledgeBase(
        x=dataset.x[indices],
        y=dataset.y[indices],
        vectors=vectors,
        input_mean=means,
        input_std=stds,
        features=dataset.features[indices],
        metadata=metadata,
        feature_names=dataset.feature_names,
    )


class HistoricalRetriever:
    METHODS = {"random", "cosine", "calendar", "hybrid", "hybrid_no_event", "event_conditioned"}

    def __init__(self, kb: KnowledgeBase, cfg: ProjectConfig):
        self.kb = kb
        self.cfg = cfg
        self.groups = {
            "weather": [i for i, name in enumerate(kb.feature_names) if name.startswith("weather_")],
            "calendar": [i for i, name in enumerate(kb.feature_names) if name.startswith("cal_")],
            "event": [i for i, name in enumerate(kb.feature_names) if name.startswith("event_")],
        }
        self.context_reference: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for group, columns in self.groups.items():
            if not columns:
                continue
            candidate = kb.features[:, columns]
            center = np.nanmean(candidate, axis=0)
            scale = np.maximum(np.nanstd(candidate, axis=0), cfg.retrieval.epsilon)
            scaled = np.nan_to_num((candidate - center) / scale)
            norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            unit = scaled / np.maximum(norms, cfg.retrieval.epsilon)
            zero = np.linalg.norm(candidate, axis=1) <= cfg.retrieval.epsilon
            self.context_reference[group] = (
                unit.astype(np.float32), center, scale, zero
            )
        try:
            self.event_context_index = kb.feature_names.index(cfg.retrieval.event_context_feature)
        except ValueError as error:
            raise ValueError(
                f"Configured event context feature is missing: {cfg.retrieval.event_context_feature}"
            ) from error
        self.candidate_event_context = (
            kb.features[:, self.event_context_index] > cfg.retrieval.epsilon
        )

    def _context_similarity(self, query_features: np.ndarray, group: str) -> np.ndarray:
        columns = self.groups[group]
        if not columns:
            return np.zeros((len(query_features), len(self.kb.metadata)), dtype=np.float32)
        query = query_features[:, columns]
        candidate_unit, center, scale, candidate_zero = self.context_reference[group]
        query_scaled = np.nan_to_num((query - center) / scale)
        query_norm = np.linalg.norm(query_scaled, axis=1, keepdims=True)
        query_unit = query_scaled / np.maximum(query_norm, self.cfg.retrieval.epsilon)
        similarity = query_unit @ candidate_unit.T
        if group == "event":
            query_zero = np.linalg.norm(query, axis=1) <= self.cfg.retrieval.epsilon
            similarity = np.clip(similarity, 0, 1)
            similarity[np.ix_(query_zero, candidate_zero)] = 1.0
            similarity[np.ix_(query_zero, ~candidate_zero)] = 0.0
            similarity[np.ix_(~query_zero, candidate_zero)] = 0.0
            return similarity.astype(np.float32)
        return np.clip((similarity + 1.0) / 2.0, 0, 1).astype(np.float32)

    def _random_selection(self, eligible: np.ndarray, count: int, window_id: str) -> np.ndarray:
        query_seed = zlib.crc32(str(window_id).encode("utf-8"))
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, query_seed]))
        return rng.choice(eligible, size=count, replace=False)

    def retrieve(
        self,
        queries: WindowDataset,
        method: str = "cosine",
        k: int | None = None,
        event_weight: float | None = None,
    ) -> RetrievalResult:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {sorted(self.METHODS)}")
        k = int(k or self.cfg.retrieval.k)
        configured_event_weight = self.cfg.retrieval.weights["event"]
        event_weight = configured_event_weight if event_weight is None else float(event_weight)
        if not 0 <= event_weight <= 1:
            raise ValueError("event_weight must be within [0, 1]")

        query_vectors, query_means, query_stds = normalize_windows(queries.x, self.cfg.retrieval.epsilon)
        n_queries = len(queries.metadata)
        horizon = queries.y.shape[1]
        prediction = np.full((n_queries, horizon), np.nan, dtype=np.float32)
        weighted_prediction = np.full_like(prediction, np.nan)
        spread = np.full_like(prediction, np.nan)
        mean_similarity = np.full(n_queries, np.nan, dtype=np.float32)
        max_similarity = np.full(n_queries, np.nan, dtype=np.float32)
        candidate_count = np.zeros(n_queries, dtype=np.int16)
        eligible_candidate_count = np.zeros(n_queries, dtype=np.int32)
        selected_event_fraction = np.full(n_queries, np.nan, dtype=np.float32)
        query_event_context = (
            queries.features[:, self.event_context_index] > self.cfg.retrieval.epsilon
        )
        event_conditioning_applied = np.zeros(n_queries, dtype=bool)
        evidence_rows: list[dict] = []

        kb_target_end = pd.to_datetime(self.kb.metadata["target_end"], utc=True).to_numpy()
        query_input_starts = pd.to_datetime(queries.metadata["input_start"], utc=True).to_numpy()
        base_weights = self.cfg.retrieval.weights
        base_total = base_weights["time_series"] + base_weights["weather"] + base_weights["calendar"]

        for block_start in range(0, n_queries, self.cfg.retrieval.block_size):
            block_end = min(block_start + self.cfg.retrieval.block_size, n_queries)
            raw_ts = query_vectors[block_start:block_end] @ self.kb.vectors.T
            ts_score = np.clip((raw_ts + 1.0) / 2.0, 0, 1)
            block_features = queries.features[block_start:block_end]
            if method in {"hybrid", "hybrid_no_event", "event_conditioned"}:
                weather = self._context_similarity(block_features, "weather")
                calendar = self._context_similarity(block_features, "calendar")
                event = self._context_similarity(block_features, "event")
            elif method == "calendar":
                weather = np.zeros_like(ts_score)
                calendar = self._context_similarity(block_features, "calendar")
                event = np.zeros_like(ts_score)
            else:
                weather = calendar = event = np.zeros_like(ts_score)

            for local_index, query_index in enumerate(range(block_start, block_end)):
                eligible = np.flatnonzero(kb_target_end < query_input_starts[query_index])
                if eligible.size == 0:
                    continue
                eligible_candidate_count[query_index] = eligible.size
                count = min(k, eligible.size)
                conditioned = False
                if method == "event_conditioned" and query_event_context[query_index]:
                    event_eligible = eligible[self.candidate_event_context[eligible]]
                    if event_eligible.size >= count:
                        eligible = event_eligible
                        conditioned = True
                    elif self.cfg.retrieval.event_stratified_fallback == "no_result":
                        continue
                event_conditioning_applied[query_index] = conditioned

                if method == "random":
                    query_row = queries.metadata.iloc[query_index]
                    selected = self._random_selection(eligible, count, query_row["window_id"])
                    selected_scores = np.zeros(count, dtype=np.float32)
                    normalized_event = np.zeros(len(self.kb.metadata), dtype=np.float32)
                    effective_event_weight = 0.0
                else:
                    if method == "cosine":
                        score_row = ts_score[local_index].copy()
                        normalized_event = np.zeros(len(self.kb.metadata), dtype=np.float32)
                        effective_event_weight = 0.0
                    elif method == "calendar":
                        score_row = calendar[local_index].copy()
                        normalized_event = np.zeros(len(self.kb.metadata), dtype=np.float32)
                        effective_event_weight = 0.0
                    else:
                        normalized_event = np.zeros(len(self.kb.metadata), dtype=np.float32)
                        if method == "hybrid_no_event":
                            normalized_event = np.zeros(len(self.kb.metadata), dtype=np.float32)
                        elif self.cfg.retrieval.normalize_event_scores and query_event_context[query_index]:
                            normalized_event[eligible] = _eligible_minmax(
                                event[local_index, eligible], self.cfg.retrieval.epsilon
                            )
                        else:
                            normalized_event = event[local_index].copy()
                        effective_event_weight = (
                            0.0
                            if method == "hybrid_no_event" or not query_event_context[query_index]
                            else event_weight
                        )
                        base_score = (
                            base_weights["time_series"] * ts_score[local_index]
                            + base_weights["weather"] * weather[local_index]
                            + base_weights["calendar"] * calendar[local_index]
                        ) / base_total
                        score_row = (
                            (1.0 - effective_event_weight) * base_score
                            + effective_event_weight * normalized_event
                        )
                    eligible_scores = score_row[eligible]
                    if count == eligible.size:
                        order = np.argsort(eligible_scores)[::-1]
                    else:
                        partition = np.argpartition(eligible_scores, -count)[-count:]
                        order = partition[np.argsort(eligible_scores[partition])[::-1]]
                    selected = eligible[order]
                    selected_scores = score_row[selected]

                normalized_future = (
                    self.kb.y[selected] - self.kb.input_mean[selected, None]
                ) / self.kb.input_std[selected, None]
                aligned = query_means[query_index] + query_stds[query_index] * normalized_future
                prediction[query_index] = aligned.mean(axis=0)
                spread[query_index] = aligned.std(axis=0)
                positive = np.maximum(selected_scores, 0)
                candidate_weights = (
                    np.full(count, 1.0 / count)
                    if positive.sum() <= self.cfg.retrieval.epsilon
                    else positive / positive.sum()
                )
                weighted_prediction[query_index] = np.average(aligned, axis=0, weights=candidate_weights)
                similarities = ts_score[local_index, selected]
                mean_similarity[query_index] = float(similarities.mean())
                max_similarity[query_index] = float(similarities.max())
                candidate_count[query_index] = count
                selected_event_fraction[query_index] = float(self.candidate_event_context[selected].mean())

                query_row = queries.metadata.iloc[query_index]
                for rank, (candidate_index, total_score) in enumerate(zip(selected, selected_scores), start=1):
                    candidate_row = self.kb.metadata.iloc[candidate_index]
                    evidence_rows.append(
                        {
                            "query_window_id": query_row["window_id"],
                            "query_origin": query_row["origin_time"],
                            "query_input_start": query_row["input_start"],
                            "method": method,
                            "rank": rank,
                            "candidate_window_id": candidate_row["window_id"],
                            "candidate_input_start": candidate_row["input_start"],
                            "candidate_origin": candidate_row["origin_time"],
                            "candidate_target_end": candidate_row["target_end"],
                            "kb_stride_hours": int(candidate_row["kb_stride_hours"]),
                            "eligible_candidate_count": int(eligible_candidate_count[query_index]),
                            "query_has_event_context": bool(query_event_context[query_index]),
                            "candidate_has_event_context": bool(self.candidate_event_context[candidate_index]),
                            "event_conditioning_applied": bool(conditioned),
                            "event_weight": float(effective_event_weight),
                            "total_score": float(total_score),
                            "time_series_score": float(ts_score[local_index, candidate_index]),
                            "weather_score": float(weather[local_index, candidate_index]),
                            "calendar_score": float(calendar[local_index, candidate_index]),
                            "event_score_raw": float(event[local_index, candidate_index]),
                            "event_score": float(normalized_event[candidate_index]),
                            "aligned_future": aligned[rank - 1].astype(float).tolist(),
                        }
                    )

        evidence = pd.DataFrame(evidence_rows)
        assert_retrieval_causality(evidence)
        return RetrievalResult(
            prediction=prediction,
            weighted_prediction=weighted_prediction,
            spread=spread,
            mean_similarity=mean_similarity,
            max_similarity=max_similarity,
            candidate_count=candidate_count,
            eligible_candidate_count=eligible_candidate_count,
            selected_event_fraction=selected_event_fraction,
            query_event_context=query_event_context,
            event_conditioning_applied=event_conditioning_applied,
            evidence=evidence,
        )


def assert_retrieval_causality(evidence: pd.DataFrame) -> None:
    if evidence.empty:
        return
    query = pd.to_datetime(evidence["query_origin"], utc=True)
    query_input_start = pd.to_datetime(evidence["query_input_start"], utc=True)
    candidate_end = pd.to_datetime(evidence["candidate_target_end"], utc=True)
    valid = (candidate_end < query_input_start) & (candidate_end < query)
    if not valid.all():
        invalid = evidence.loc[~valid].head()
        raise AssertionError(f"Retrieval leakage detected:\n{invalid}")
