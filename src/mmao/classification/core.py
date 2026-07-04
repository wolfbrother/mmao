from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .datasets import load_builtin_dataset


@dataclass(frozen=True)
class ClassificationProblem:
    name: str
    dataset_name: str
    classifier_name: str = "svm_rbf"
    metric: str = "balanced_accuracy"

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return load_builtin_dataset(self.dataset_name)


@dataclass
class MMAOClassificationConfig:
    iterations: int = 28
    seed: int = 3
    initial_agents: int = 10
    minimum_agents: int = 6
    maximum_agents: int = 18
    communal_share: float = 0.28
    maintenance_cost: float = 0.04
    offspring_cost: float = 0.40
    feature_flip_scale: float = 0.16
    param_step_scale: float = 0.16
    feature_penalty: float = 0.08
    social_pull: float = 0.22
    prior_pull: float = 0.20


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=seed + 101
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def metric_score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"Unsupported metric: {metric}")


def build_estimator(classifier_name: str, params: np.ndarray):
    if classifier_name == "svm_rbf":
        c = float(10 ** np.clip(params[0], -2.0, 2.5))
        gamma = float(10 ** np.clip(params[1], -4.0, 1.0))
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=c, gamma=gamma, kernel="rbf"))])
    if classifier_name == "knn":
        n_neighbors = int(np.clip(round(1 + (params[0] + 2.0) / 4.0 * 14), 1, 15))
        weights = "distance" if params[1] > 0.0 else "uniform"
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))])
    if classifier_name == "logreg":
        c = float(10 ** np.clip(params[0], -3.0, 2.0))
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=1000))])
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def decode(candidate: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    feature_logits = candidate[:n_features]
    mask = feature_logits > 0.0
    if not np.any(mask):
        mask[np.argmax(feature_logits)] = True
    return mask, candidate[n_features:]


def compute_feature_priors(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    priors = mutual_info_classif(x_train, y_train, random_state=0)
    if float(np.max(priors)) <= 1e-12:
        return np.zeros_like(priors)
    return np.asarray(priors / np.max(priors), dtype=float)


def derive_feature_budget(feature_priors: np.ndarray) -> tuple[int, int]:
    n_features = feature_priors.size
    if n_features <= 4:
        return 1, max(1, n_features // 2)
    total = float(np.sum(feature_priors))
    if total <= 1e-12:
        base = max(2, int(round(math.sqrt(n_features))))
        return min(base, n_features), min(base + max(1, n_features // 8), n_features)
    ranking = np.argsort(feature_priors)[::-1]
    cumulative = np.cumsum(feature_priors[ranking]) / total
    min_count = int(np.searchsorted(cumulative, 0.68) + 1)
    target_count = int(np.searchsorted(cumulative, 0.90) + 1)
    floor = max(2, int(round(math.sqrt(n_features) * 1.1)))
    min_count = max(min_count, floor)
    target_count = max(target_count, min_count + 1)
    return min(min_count, n_features), min(target_count, n_features)


def repair_mask(feature_logits: np.ndarray, feature_priors: np.ndarray, minimum_count: int) -> np.ndarray:
    mask = feature_logits > 0.0
    if int(mask.sum()) >= minimum_count:
        return mask
    ranking = np.argsort(feature_logits + 0.75 * feature_priors)[::-1]
    repaired = np.zeros_like(mask, dtype=bool)
    repaired[ranking[:minimum_count]] = True
    return repaired


def sparsity_penalty(feature_ratio: float, base_penalty: float, target_ratio: float) -> float:
    excess = max(0.0, feature_ratio - target_ratio)
    return 0.85 * base_penalty * excess + 0.12 * base_penalty * excess * excess


def evaluate_candidate(
    candidate: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
    classifier_name: str,
    feature_penalty: float,
    target_ratio: float,
    minimum_feature_count: int,
    feature_priors: np.ndarray,
) -> dict[str, float | np.ndarray]:
    n_features = x_train.shape[1]
    raw_mask, params = decode(candidate, n_features)
    mask = repair_mask(candidate[:n_features], feature_priors, minimum_feature_count) if int(raw_mask.sum()) < minimum_feature_count else raw_mask
    estimator = build_estimator(classifier_name, params)
    estimator.fit(x_train[:, mask], y_train)
    train_pred = estimator.predict(x_train[:, mask])
    val_pred = estimator.predict(x_val[:, mask])
    train_metric = metric_score(metric, y_train, train_pred)
    val_metric = metric_score(metric, y_val, val_pred)
    feature_ratio = float(mask.sum() / n_features)
    overfit_gap = max(0.0, train_metric - val_metric)
    penalty = sparsity_penalty(feature_ratio, feature_penalty, target_ratio)
    objective = val_metric - penalty - 0.18 * overfit_gap
    return {
        "objective": objective,
        "metric_value": val_metric,
        "feature_ratio": feature_ratio,
        "selected_features": float(mask.sum()),
        "mask": mask.astype(int),
        "train_metric": train_metric,
        "overfit_gap": overfit_gap,
    }


def evaluate_final_model(
    candidate: np.ndarray,
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metric: str,
    classifier_name: str,
) -> dict[str, float]:
    mask, params = decode(candidate, x_trainval.shape[1])
    estimator = build_estimator(classifier_name, params)
    estimator.fit(x_trainval[:, mask], y_trainval)
    y_pred = estimator.predict(x_test[:, mask])
    return {
        "test_metric": metric_score(metric, y_test, y_pred),
        "test_feature_ratio": float(mask.sum() / x_trainval.shape[1]),
        "test_selected_features": float(mask.sum()),
    }


def consensus_from_elites(elites: list[dict[str, object]], n_features: int) -> np.ndarray:
    if not elites:
        return np.zeros(n_features, dtype=float)
    masks = [np.asarray(elite["mask"], dtype=float) for elite in elites]
    return np.mean(np.vstack(masks), axis=0)


def elite_refine(
    candidate: np.ndarray,
    outcome: dict[str, float | np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
    classifier_name: str,
    feature_penalty: float,
    target_ratio: float,
    minimum_feature_count: int,
    feature_priors: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
    n_features = x_train.shape[1]
    best_candidate = candidate.copy()
    best_outcome = outcome
    mask = np.asarray(outcome["mask"], dtype=int)
    selected = np.where(mask > 0)[0].tolist()
    unselected = np.where(mask == 0)[0].tolist()

    if len(selected) > minimum_feature_count:
        ranked_drop = sorted(selected, key=lambda idx: feature_priors[idx])
        for idx in ranked_drop[: min(2, len(ranked_drop))]:
            trial = best_candidate.copy()
            trial[idx] = -abs(trial[idx]) - 0.1
            trial_outcome = evaluate_candidate(
                trial, x_train, y_train, x_val, y_val, metric, classifier_name, feature_penalty, target_ratio, minimum_feature_count, feature_priors
            )
            if float(trial_outcome["objective"]) >= float(best_outcome["objective"]) - 1e-5:
                best_candidate = trial
                best_outcome = trial_outcome

    if unselected:
        ranked_add = sorted(unselected, key=lambda idx: feature_priors[idx], reverse=True)
        for idx in ranked_add[: min(2, len(ranked_add))]:
            trial = best_candidate.copy()
            trial[idx] = abs(trial[idx]) + 0.15
            trial_outcome = evaluate_candidate(
                trial, x_train, y_train, x_val, y_val, metric, classifier_name, feature_penalty, target_ratio, minimum_feature_count, feature_priors
            )
            if float(trial_outcome["objective"]) > float(best_outcome["objective"]):
                best_candidate = trial
                best_outcome = trial_outcome
    return best_candidate, best_outcome


def initialize_population(rng: np.random.Generator, n_agents: int, n_dims: int, feature_priors: np.ndarray) -> list[dict[str, np.ndarray | float]]:
    n_features = feature_priors.size
    agents: list[dict[str, np.ndarray | float]] = []
    for _ in range(n_agents):
        position = rng.normal(0.0, 0.85, size=n_dims)
        position[:n_features] += (feature_priors - 0.5) * 0.9
        agents.append(
            {
                "position": position,
                "energy": 1.0,
                "role": rng.uniform(0.15, 0.85),
                "best_position": position.copy(),
                "best_objective": -math.inf,
                "mask": np.zeros(n_features, dtype=int),
            }
        )
    return agents


def mutate_position(
    rng: np.random.Generator,
    position: np.ndarray,
    role: float,
    feature_priors: np.ndarray,
    consensus: np.ndarray,
    anchor: np.ndarray,
    config: MMAOClassificationConfig,
) -> np.ndarray:
    n_features = feature_priors.size
    candidate = position.copy()
    candidate[:n_features] += (
        rng.normal(0.0, config.feature_flip_scale * (1.15 - role), size=n_features)
        + config.prior_pull * (feature_priors - 0.5)
        + config.social_pull * (consensus - 0.5)
        + 0.15 * (anchor[:n_features] - candidate[:n_features])
    )
    candidate[n_features:] += (
        rng.normal(0.0, config.param_step_scale * (0.55 + role), size=candidate.size - n_features)
        + 0.30 * role * (anchor[n_features:] - candidate[n_features:])
    )
    return candidate


def optimize_classification_problem(problem: ClassificationProblem, config: MMAOClassificationConfig) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    x, y = problem.load()
    param_dims = 2 if problem.classifier_name != "logreg" else 1
    total_dims = x.shape[1] + param_dims
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y, config.seed)
    x_trainval = np.concatenate([x_train, x_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    feature_priors = compute_feature_priors(x_train, y_train)
    minimum_feature_count, target_feature_count = derive_feature_budget(feature_priors)
    target_ratio = target_feature_count / x.shape[1]
    agents = initialize_population(rng, config.initial_agents, total_dims, feature_priors)
    communal_budget = 1.0
    gain_scale = 1.0
    history: list[dict[str, float]] = []
    archive: list[dict[str, object]] = []
    global_best: dict[str, object] | None = None

    for iteration in range(config.iterations):
        archive = sorted(archive, key=lambda item: float(item["objective"]), reverse=True)[: max(2, config.minimum_agents // 2)]
        consensus = consensus_from_elites(archive, x.shape[1])
        objectives: list[float] = []

        for agent in agents:
            anchor = np.asarray(agent["best_position"]) if global_best is None else np.asarray(global_best["position"])
            candidate = mutate_position(rng, np.asarray(agent["position"]), float(agent["role"]), feature_priors, consensus, anchor, config)
            outcome = evaluate_candidate(
                candidate,
                x_train,
                y_train,
                x_val,
                y_val,
                problem.metric,
                problem.classifier_name,
                config.feature_penalty,
                target_ratio,
                minimum_feature_count,
                feature_priors,
            )
            candidate, outcome = elite_refine(
                candidate,
                outcome,
                x_train,
                y_train,
                x_val,
                y_val,
                problem.metric,
                problem.classifier_name,
                config.feature_penalty,
                target_ratio,
                minimum_feature_count,
                feature_priors,
            )

            objective = float(outcome["objective"])
            previous = float(agent["best_objective"])
            gain = max(0.0, objective - previous) if previous > -math.inf else max(0.0, objective)
            gain_scale = 0.9 * gain_scale + 0.1 * max(gain, 1e-6)
            normalized_gain = gain / max(gain_scale, 1e-6)
            reward = math.tanh(2.1 * normalized_gain)

            agent["position"] = candidate
            agent["mask"] = np.asarray(outcome["mask"])
            agent["energy"] = float(np.clip(float(agent["energy"]) + (1.0 - config.communal_share) * reward - config.maintenance_cost, 0.0, 2.5))
            communal_budget = max(0.0, communal_budget + config.communal_share * reward)
            stability = 1.0 - min(1.0, float(outcome["overfit_gap"]) * 2.0)
            agent["role"] = float(
                np.clip(
                    0.78 * float(agent["role"])
                    + 0.08 * stability
                    + 0.08 * min(1.0, float(outcome["feature_ratio"]) / max(target_ratio, 1e-6))
                    + 0.06 * (1.0 - float(outcome["feature_ratio"])),
                    0.05,
                    0.95,
                )
            )

            if objective > previous:
                agent["best_objective"] = objective
                agent["best_position"] = candidate.copy()
                archive.append({"objective": objective, "position": candidate.copy(), "mask": np.asarray(outcome["mask"]).copy()})
                if global_best is None or objective > float(global_best["objective"]):
                    global_best = {
                        "objective": objective,
                        "position": candidate.copy(),
                        "metric_value": float(outcome["metric_value"]),
                        "feature_ratio": float(outcome["feature_ratio"]),
                        "selected_features": float(outcome["selected_features"]),
                        "mask": np.asarray(outcome["mask"]).copy(),
                    }
            objectives.append(objective)

        agents = [agent for agent in agents if float(agent["energy"]) > 0.05]
        if len(agents) < config.minimum_agents:
            agents.extend(initialize_population(rng, config.minimum_agents - len(agents), total_dims, feature_priors))

        if communal_budget > config.offspring_cost and len(agents) < config.maximum_agents and global_best is not None:
            child = np.asarray(global_best["position"]).copy()
            child[: x.shape[1]] += 0.08 * (feature_priors - 0.5) + rng.normal(0.0, 0.06, size=x.shape[1])
            child[x.shape[1] :] += rng.normal(0.0, 0.08, size=param_dims)
            agents.append(
                {
                    "position": child,
                    "energy": 0.85,
                    "role": 0.50,
                    "best_position": child.copy(),
                    "best_objective": -math.inf,
                    "mask": np.zeros(x.shape[1], dtype=int),
                }
            )
            communal_budget -= config.offspring_cost

        best_objective = max(objectives) if objectives else (float(global_best["objective"]) if global_best is not None else -math.inf)
        mean_feature_ratio = float(np.mean([np.mean(np.asarray(agent["mask"])) for agent in agents])) if agents else 0.0
        history.append(
            {
                "iteration": iteration,
                "best_objective": best_objective,
                "population": float(len(agents)),
                "communal_budget": float(communal_budget),
                "mean_feature_ratio": mean_feature_ratio,
                "target_feature_ratio": target_ratio,
            }
        )

    if global_best is None:
        raise RuntimeError("Optimization produced no valid global best.")

    final_eval = evaluate_final_model(
        np.asarray(global_best["position"]),
        x_trainval,
        y_trainval,
        x_test,
        y_test,
        problem.metric,
        problem.classifier_name,
    )
    return {"problem": problem.name, "history": history, **global_best, **final_eval}
