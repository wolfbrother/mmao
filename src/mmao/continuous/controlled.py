# Copyright (c) 2026 Jinliang Xu
# Author: Jinliang Xu
# Email: jlxufly@gmail.com and xujinliang@caict.ac.cn
# SPDX-License-Identifier: Apache-2.0
"""Common-kernel research controllers with explicit resource accounting.

This additive research API has its own version identity. Existing public
optimizers keep their established implementation and configuration semantics.
"""

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np

from .benchmarks import ContinuousProblem
from ..common.validation import finite_number, integer


@dataclass(frozen=True)
class ResourceControlConfig:
    policy: str = "M"
    kernel: str = "probe"
    seed: int = 7
    max_evaluations: int = 1000
    initial_agents: int = 8
    minimum_agents: int = 4
    maximum_agents: int = 12
    learning_rate: float = 0.18
    initial_step: float = 0.04
    success_target: float = 0.2
    donation: float = 0.25
    maintenance: float = 0.14
    initial_energy: float = 6.0
    maximum_energy: float = 16.0
    initial_pool: float = 2.0
    temperature: float = 2.0
    exploration: float = 0.1
    reinvestment: float = 0.25
    turnover_period: int = 8

    def validate(self):
        if self.policy not in {"F", "I", "M", "D"}:
            raise ValueError("policy must be F, I, M or D.")
        if self.kernel not in {"probe", "gaussian"}:
            raise ValueError("kernel must be probe or gaussian.")
        for name in ("initial_agents", "minimum_agents", "maximum_agents", "turnover_period"):
            integer(name, getattr(self, name))
        integer("seed", self.seed, 0)
        integer("max_evaluations", self.max_evaluations, self.initial_agents)
        if not self.minimum_agents <= self.initial_agents <= self.maximum_agents:
            raise ValueError("Invalid population envelope.")
        for name in ("learning_rate", "initial_step", "success_target", "donation", "exploration", "reinvestment"):
            value = getattr(self, name)
            finite_number(name, value)
            if not 0 < value < 1:
                raise ValueError(f"{name} must lie in (0, 1).")
        for name in ("maintenance", "initial_energy", "maximum_energy", "temperature"):
            finite_number(name, getattr(self, name))
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        finite_number("initial_pool", self.initial_pool)
        if self.initial_pool < 0 or self.initial_energy > self.maximum_energy:
            raise ValueError("Invalid energy or pool initialization.")


def derive_seed(seed: int, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}|{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def allocation_probabilities(energy, exploration: float, temperature: float):
    """Softmax resource weights mixed with a uniform exploration floor."""
    values = np.asarray(energy, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Energy must be a nonempty finite vector.")
    if not 0 < exploration < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid allocation parameters.")
    weights = np.exp((values - values.max()) / temperature)
    return exploration / len(values) + (1 - exploration) * weights / weights.sum()


def optimize_resource_controlled(problem: ContinuousProblem, config: ResourceControlConfig):
    """Execute version resource-control-v1 with a shared proposal kernel.

    Complete three-evaluation actions are used by the probe kernel. Gaussian
    proposals use one call. Every candidate, including initial and newborn
    agents, passes through the same counter. The returned evaluation trace
    contains actual observed values, while accounting rows describe each round.
    """
    config.validate()
    initialization = np.random.default_rng(derive_seed(config.seed, "initialization"))
    proposals = np.random.default_rng(derive_seed(config.seed, "proposal"))
    control = np.random.default_rng(derive_seed(config.seed, "control"))
    lower = np.array([pair[0] for pair in problem.bounds])
    upper = np.array([pair[1] for pair in problem.bounds])
    span = upper - lower
    agents = []
    evaluations = []
    rounds = []
    best_value = math.inf
    best_position = None
    pool = config.initial_pool
    success_ema = config.success_target
    recent_returns = []
    action_cost = 3 if config.kernel == "probe" else 1

    def evaluate(position):
        nonlocal best_value, best_position
        if len(evaluations) >= config.max_evaluations:
            raise RuntimeError("Evaluation budget exceeded.")
        value = problem.evaluate(position.tolist())
        if not math.isfinite(value):
            raise ValueError("Objective returned a nonfinite value.")
        if value < best_value:
            best_value = value
            best_position = position.copy()
        evaluations.append((len(evaluations) + 1, best_value))
        return value

    def new_agent(position):
        return {
            "position": position, "fitness": evaluate(position),
            "energy": config.initial_energy, "role": 0.5,
            "step": config.initial_step, "success": config.success_target,
            "stagnation": 0,
        }

    for _ in range(config.initial_agents):
        agents.append(new_agent(initialization.uniform(lower, upper)))

    def action(agent, anchor, scale):
        current = agent["position"]
        role = agent["role"]
        radius = agent["step"] * span * (1.25 - role)
        candidates = []
        if config.kernel == "probe":
            direction = proposals.normal(size=problem.dimension)
            direction /= max(np.linalg.norm(direction), 1e-12)
            plus = np.clip(current + radius * direction, lower, upper)
            minus = np.clip(current - radius * direction, lower, upper)
            candidates = [(evaluate(plus), plus), (evaluate(minus), minus)]
            probe = min(candidates, key=lambda item: item[0])[1]
            offset = 0.5 * (probe - current)
        else:
            offset = np.zeros(problem.dimension)
        candidate = np.clip(
            current + 0.25 * role * (anchor - current) + offset
            + radius * proposals.normal(size=problem.dimension), lower, upper,
        )
        candidates.append((evaluate(candidate), candidate))
        value, position = min(candidates, key=lambda item: item[0])
        gain = max(0.0, agent["fitness"] - value)
        success = float(gain > 0)
        if success:
            agent["position"] = position
            agent["fitness"] = value
            agent["stagnation"] = 0
        else:
            agent["stagnation"] += 1
        reward = min(4.0, math.log1p(gain / scale))
        agent["success"] += config.learning_rate * (success - agent["success"])
        recent_returns.append(gain)
        return reward, success

    iteration = 0
    while config.max_evaluations - len(evaluations) >= action_cost:
        iteration += 1
        pool_before = pool
        donations = recovered = spent = born_cost = 0.0
        anchor = best_position.copy()
        # Scale uses observed objective changes, independent of the known optimum.
        positive = [value for value in recent_returns[-24:] if value > 0]
        scale = float(np.median(positive)) if positive else max(1.0, abs(best_value))
        scale = max(scale, 1e-12)
        rewards = []
        for agent in list(agents):
            if config.max_evaluations - len(evaluations) < action_cost:
                break
            reward, success = action(agent, anchor, scale)
            rewards.append(success)
            if config.policy in {"M", "D"}:
                contribution = config.donation * reward
                donations += contribution
                pool += contribution
                agent["energy"] = float(np.clip(
                    agent["energy"] + reward - contribution - config.maintenance,
                    0.0, config.maximum_energy,
                ))
                pressure = agent["energy"] / config.maximum_energy - 0.5
                role_signal = success - config.success_target + 0.2 * pressure
                scale_signal = success - config.success_target + 0.1 * pressure
            else:
                role_signal = agent["success"] - config.success_target
                scale_signal = success - config.success_target
            if config.policy != "F":
                agent["role"] = float(np.clip(agent["role"] + config.learning_rate * role_signal, 0, 1))
                agent["step"] = float(np.clip(
                    agent["step"] * math.exp(config.learning_rate * scale_signal), 1e-5, 0.3,
                ))

        success_ema += config.learning_rate * (float(np.mean(rewards)) - success_ema)
        if config.policy in {"M", "D"}:
            for agent in list(agents):
                if len(agents) > config.minimum_agents and agent["energy"] <= 0.05 and agent["stagnation"] >= config.turnover_period:
                    recovered += 0.15 * agent["energy"]
                    pool += 0.15 * agent["energy"]
                    agents[:] = [other for other in agents if other is not agent]
            allowance = min(len(agents), int(config.reinvestment * pool))
            probabilities = np.full(len(agents), 1 / len(agents))
            if config.policy == "M":
                probabilities = allocation_probabilities(
                    [agent["energy"] for agent in agents], config.exploration, config.temperature,
                )
            # Weights and allowance are frozen before extra evaluations this round.
            for _ in range(allowance):
                if config.max_evaluations - len(evaluations) < action_cost:
                    break
                selected = int(control.choice(len(agents), p=probabilities))
                recipient = agents[selected]
                reward, _ = action(recipient, anchor, scale)
                contribution = config.donation * reward
                donations += contribution
                pool += contribution
                recipient["energy"] = float(np.clip(
                    recipient["energy"] + reward - contribution - config.maintenance,
                    0.0, config.maximum_energy,
                ))
                pool -= 1.0
                spent += 1.0
            target = len(agents)
            if pool >= config.initial_energy and target < config.maximum_agents:
                target += 1
        elif config.policy == "I":
            target = int(round(config.minimum_agents + (config.maximum_agents - config.minimum_agents) * success_ema))
        else:
            target = config.initial_agents

        while len(agents) > target:
            worst = max(range(len(agents)), key=lambda index: agents[index]["fitness"])
            agents.pop(worst)
        while len(agents) < target and len(evaluations) < config.max_evaluations:
            position = np.clip(anchor + config.initial_step * span * proposals.normal(size=problem.dimension), lower, upper)
            agents.append(new_agent(position))
            if config.policy in {"M", "D"}:
                pool -= config.initial_energy
                born_cost += config.initial_energy

        refresh_due = config.policy == "F" and iteration % config.turnover_period == 0
        if config.policy == "I":
            refresh_due = max(agent["stagnation"] for agent in agents) >= config.turnover_period
        if refresh_due and len(evaluations) < config.max_evaluations:
            worst = max(range(len(agents)), key=lambda index: agents[index]["stagnation"])
            agents[worst] = new_agent(proposals.uniform(lower, upper))
        residual = pool - (pool_before + donations + recovered - spent - born_cost)
        if abs(residual) > 1e-9 or pool < -1e-9:
            raise RuntimeError("Resource accounting invariant failed.")
        rounds.append({
            "iteration": iteration, "nfev": len(evaluations), "population": len(agents),
            "pool_before": pool_before, "donations": donations, "recovered": recovered,
            "reinvestment_cost": spent, "birth_cost": born_cost, "pool": pool,
            "ledger_residual": residual, "mean_energy": float(np.mean([a["energy"] for a in agents])),
            "mean_role": float(np.mean([a["role"] for a in agents])), "success_ema": success_ema,
        })
    return {
        "implementation": "resource-control-v1", "config": asdict(config),
        "best_fitness": best_value, "best_position": best_position.tolist(),
        "nfev": len(evaluations), "evaluations": evaluations, "history": rounds,
    }
