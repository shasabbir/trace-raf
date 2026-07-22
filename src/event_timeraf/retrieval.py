from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .windows import WindowDataset


def normalize_windows(values: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    scale = np.nanstd(candidates, axis=0)
    scale = np.maximum(scale, epsilon)
    candidate_scaled = np.nan_to_num((candidates - center) / scale)
    query_scaled = np.nan_to_num((queries - center) / scale)
    candidate_norm = np.linalg.norm(candidate_scaled, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_scaled, axis=1, keepdims=True)
    candidate_unit = candidate_scaled / np.maximum(candidate_norm, epsilon)
    query_unit = query_scaled / np.maximum(query_norm, epsilon)
    return candidate_unit.astype(np.float32), query_unit.astype(np.float32)


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
            ]
        )


def build_knowledge_base(dataset: WindowDataset, cfg: ProjectConfig) -> KnowledgeBase:
    train_indices = np.flatnonzero(dataset.metadata["split"].to_numpy() == "train")
    if train_indices.size == 0:
        raise ValueError("Training split is empty")
    origin_rows = dataset.metadata.loc[train_indices, "origin_row"].to_numpy(dtype=int)
    keep = origin_rows % cfg.retrieval.kb_stride_hours == 0
    indices = train_indices[keep]
    if indices.size < cfg.retrieval.k:
        raise ValueError("Knowledge base has fewer candidates than configured k")
    metadata = dataset.metadata.loc[indices].reset_index(drop=True)
    if len(metadata) > 1:
        candidate_end = pd.to_datetime(metadata["target_end"], utc=True).to_numpy()
        next_input_start = pd.to_datetime(metadata["input_start"], utc=True).to_numpy()[1:]
        if not (candidate_end[:-1] < next_input_start).all():
            raise AssertionError(
                "Knowledge-base windows overlap; increase retrieval.kb_stride_hours"
            )
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
    def __init__(self, kb: KnowledgeBase, cfg: ProjectConfig):
        self.kb = kb
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.groups = {
            "weather": [i for i, name in enumerate(kb.feature_names) if name.startswith("weather_")],
            "calendar": [i for i, name in enumerate(kb.feature_names) if name.startswith("cal_")],
            "event": [i for i, name in enumerate(kb.feature_names) if name.startswith("event_")],
        }

    def _context_similarity(self, query_features: np.ndarray, group: str) -> np.ndarray:
        columns = self.groups[group]
        if not columns:
            return np.zeros((len(query_features), len(self.kb.metadata)), dtype=np.float32)
        candidate = self.kb.features[:, columns]
        query = query_features[:, columns]
        candidate_unit, query_unit = _standardized_unit(candidate, query, self.cfg.retrieval.epsilon)
        similarity = query_unit @ candidate_unit.T
        if group == "event":
            candidate_zero = np.linalg.norm(candidate, axis=1) <= self.cfg.retrieval.epsilon
            query_zero = np.linalg.norm(query, axis=1) <= self.cfg.retrieval.epsilon
            similarity = np.clip(similarity, 0, 1)
            similarity[np.ix_(query_zero, candidate_zero)] = 1.0
            similarity[np.ix_(query_zero, ~candidate_zero)] = 0.0
            similarity[np.ix_(~query_zero, candidate_zero)] = 0.0
            return similarity.astype(np.float32)
        return np.clip((similarity + 1.0) / 2.0, 0, 1).astype(np.float32)

    def retrieve(
        self,
        queries: WindowDataset,
        method: str = "cosine",
        k: int | None = None,
    ) -> RetrievalResult:
        if method not in {"random", "cosine", "hybrid", "hybrid_no_event"}:
            raise ValueError("method must be random, cosine, hybrid, or hybrid_no_event")
        k = k or self.cfg.retrieval.k
        query_vectors, query_means, query_stds = normalize_windows(queries.x, self.cfg.retrieval.epsilon)
        n_queries = len(queries.metadata)
        horizon = queries.y.shape[1]
        prediction = np.full((n_queries, horizon), np.nan, dtype=np.float32)
        weighted_prediction = np.full_like(prediction, np.nan)
        spread = np.full_like(prediction, np.nan)
        mean_similarity = np.full(n_queries, np.nan, dtype=np.float32)
        max_similarity = np.full(n_queries, np.nan, dtype=np.float32)
        candidate_count = np.zeros(n_queries, dtype=np.int16)
        evidence_rows: list[dict] = []

        kb_target_end = pd.to_datetime(self.kb.metadata["target_end"], utc=True).to_numpy()
        query_input_starts = pd.to_datetime(queries.metadata["input_start"], utc=True).to_numpy()
        weights = self.cfg.retrieval.weights

        for block_start in range(0, n_queries, self.cfg.retrieval.block_size):
            block_end = min(block_start + self.cfg.retrieval.block_size, n_queries)
            block_vectors = query_vectors[block_start:block_end]
            raw_ts = block_vectors @ self.kb.vectors.T
            ts_score = np.clip((raw_ts + 1.0) / 2.0, 0, 1)
            if method in {"hybrid", "hybrid_no_event"}:
                block_features = queries.features[block_start:block_end]
                weather = self._context_similarity(block_features, "weather")
                calendar = self._context_similarity(block_features, "calendar")
                if method == "hybrid":
                    event = self._context_similarity(block_features, "event")
                    scores = (
                        weights["time_series"] * ts_score
                        + weights["weather"] * weather
                        + weights["calendar"] * calendar
                        + weights["event"] * event
                    )
                else:
                    event = np.zeros_like(ts_score)
                    denominator = weights["time_series"] + weights["weather"] + weights["calendar"]
                    scores = (
                        weights["time_series"] * ts_score
                        + weights["weather"] * weather
                        + weights["calendar"] * calendar
                    ) / denominator
            else:
                weather = calendar = event = np.zeros_like(ts_score)
                scores = ts_score.copy()

            for local_index, query_index in enumerate(range(block_start, block_end)):
                # The complete candidate input and target must precede the
                # query lookback. This prevents near-duplicate retrieval from
                # reusing values already present in the query input.
                eligible = np.flatnonzero(kb_target_end < query_input_starts[query_index])
                if eligible.size == 0:
                    continue
                count = min(k, eligible.size)
                if method == "random":
                    selected = self.rng.choice(eligible, size=count, replace=False)
                    selected_scores = np.zeros(count, dtype=float)
                else:
                    eligible_scores = scores[local_index, eligible]
                    if count == eligible.size:
                        order = np.argsort(eligible_scores)[::-1]
                    else:
                        partition = np.argpartition(eligible_scores, -count)[-count:]
                        order = partition[np.argsort(eligible_scores[partition])[::-1]]
                    selected = eligible[order]
                    selected_scores = scores[local_index, selected]

                normalized_future = (
                    self.kb.y[selected] - self.kb.input_mean[selected, None]
                ) / self.kb.input_std[selected, None]
                aligned = query_means[query_index] + query_stds[query_index] * normalized_future
                prediction[query_index] = aligned.mean(axis=0)
                spread[query_index] = aligned.std(axis=0)
                positive = np.maximum(selected_scores, 0)
                if positive.sum() <= self.cfg.retrieval.epsilon:
                    candidate_weights = np.full(count, 1.0 / count)
                else:
                    candidate_weights = positive / positive.sum()
                weighted_prediction[query_index] = np.average(aligned, axis=0, weights=candidate_weights)
                similarities = ts_score[local_index, selected]
                mean_similarity[query_index] = float(similarities.mean())
                max_similarity[query_index] = float(similarities.max())
                candidate_count[query_index] = count

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
                            "total_score": float(total_score),
                            "time_series_score": float(ts_score[local_index, candidate_index]),
                            "weather_score": float(weather[local_index, candidate_index]),
                            "calendar_score": float(calendar[local_index, candidate_index]),
                            "event_score": float(event[local_index, candidate_index]),
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
