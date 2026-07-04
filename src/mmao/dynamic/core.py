from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ..common import clamp, median_upper_scale
from ..continuous.core import (
    MMAOContinuousConfig,
    clip_position,
    init_agent,
    mean_position,
    optimize_continuous_problem,
    random_unit_vector,
)
from ..continuous.benchmarks import ContinuousProblem, rastrigin, sphere
from .benchmarks import DynamicScenario


Vector = list[float]


@dataclass(frozen=True)
class DynamicOptimizationProblem:
    name: str
    scenario: DynamicScenario
    bounds: list[tuple[float, float]]
    optimum_value: float = 0.0


@dataclass
class MMAODynamicConfig:
    iterations: int = 180
    seed: int = 7
    initial_agents: int = 8
    minimum_agents: int = 4
    maximum_agents: int = 24
    initial_energy: float = 8.0
    max_energy: float = 18.0
    success_feedback_rate: float = 0.18
    role_update_rate: float = 0.16
    change_response_rate: float = 0.28
    reinvestment_bias: float = 0.58
    memory_refresh_rate: float = 0.55
    recovery_exploration_boost: float = 0.22
    recovery_window: int = 6
    scout_injection_ratio: float = 0.20


def ackley(vector: Vector) -> float:
    dimension = len(vector)
    mean_square = sum(value * value for value in vector) / max(1, dimension)
    mean_cosine = sum(math.cos(2.0 * math.pi * value) for value in vector) / max(1, dimension)
    return -20.0 * math.exp(-0.2 * math.sqrt(mean_square)) - math.exp(mean_cosine) + 20.0 + math.e


def next_shift(rng: random.Random, dimension: int, previous: Vector, severity: float, shift_scale: float) -> Vector:
    direction = random_unit_vector(rng, dimension)
    magnitude = shift_scale * (0.65 + 0.35 * severity)
    candidate = [old + magnitude * delta for old, delta in zip(previous, direction)]
    return [clamp(value, -3.6, 3.6) for value in candidate]


def evaluate_dynamic(problem_name: str, position: Vector, shift: Vector) -> float:
    centered = [value - delta for value, delta in zip(position, shift)]
    lowered = problem_name.lower()
    if lowered.startswith("dyn-sphere"):
        return sphere(centered)
    if lowered.startswith("dyn-ackley"):
        return ackley(centered)
    return rastrigin(centered)


def optimize_dynamic_problem(problem: DynamicOptimizationProblem, config: MMAODynamicConfig) -> dict[str, object]:
    rng = random.Random(config.seed)
    dimensions = problem.scenario.dimension
    spans = [upper - lower for lower, upper in problem.bounds]

    static_problem = ContinuousProblem(
        name=f"{problem.name}-static-proxy",
        dimension=dimensions,
        bounds=problem.bounds,
        objective=lambda position: evaluate_dynamic(problem.name, position, [0.0] * dimensions),
    )
    static_config = MMAOContinuousConfig(
        iterations=config.iterations,
        seed=config.seed,
        initial_agents=config.initial_agents,
        minimum_agents=config.minimum_agents,
        maximum_agents=config.maximum_agents,
        initial_energy=config.initial_energy,
        max_energy=config.max_energy,
        role_feedback_rate=config.role_update_rate,
    )

    population = [
        init_agent(static_problem.objective, static_problem.bounds, static_config, rng)
        for _ in range(static_config.initial_agents)
    ]

    shift = [0.0] * dimensions
    regime_id = 0
    best_position = list(population[0].position)
    best_fitness = math.inf
    success_rate = 0.5
    resource_pool = 0.85 * static_config.initial_agents * static_config.initial_energy
    recent_gains: list[float] = []
    last_change_gap = 0.0
    response_rounds_remaining = 0
    offline_errors: list[float] = []
    history: list[dict[str, float | int]] = []

    def reevaluate_population(current_shift: Vector) -> None:
        nonlocal best_position, best_fitness
        best_fitness = math.inf
        for agent in population:
            agent.fitness = evaluate_dynamic(problem.name, agent.position, current_shift)
            memory_fit = evaluate_dynamic(problem.name, agent.best_position, current_shift)
            if memory_fit < agent.fitness:
                agent.position = list(agent.best_position)
                agent.fitness = memory_fit
            else:
                agent.best_position = list(agent.position)
                agent.best_fitness = agent.fitness
            if agent.fitness < best_fitness:
                best_fitness = agent.fitness
                best_position = list(agent.position)

    reevaluate_population(shift)
    history.append(
        {
            "iteration": 0,
            "best_fitness": round(best_fitness, 6),
            "offline_error": round(best_fitness - problem.optimum_value, 6),
            "population": len(population),
            "resource_pool": round(resource_pool, 6),
            "success_rate": round(success_rate, 6),
            "mean_role": round(sum(agent.role for agent in population) / len(population), 6),
            "regime_id": regime_id,
            "change_gap": 0.0,
        }
    )

    for iteration in range(1, config.iterations + 1):
        if iteration > 1 and (iteration - 1) % problem.scenario.change_frequency == 0:
            pre_change_best = best_fitness
            shift = next_shift(
                rng,
                dimensions,
                shift,
                problem.scenario.change_severity,
                problem.scenario.shift_scale,
            )
            regime_id += 1
            reevaluate_population(shift)
            change_gap = max(0.0, best_fitness - pre_change_best)
            last_change_gap = change_gap
            response_rounds_remaining = config.recovery_window
            resource_pool += config.change_response_rate * static_config.initial_agents * static_config.initial_energy
            scarcity = clamp(change_gap / max(1.0, abs(pre_change_best) + 1.0), 0.0, 1.0)
            for agent in population:
                agent.stagnation = 0
                agent.energy = min(
                    static_config.max_energy,
                    agent.energy + 0.18 * resource_pool / max(1, len(population)),
                )
                agent.role = clamp(
                    agent.role
                    - (0.25 + config.recovery_exploration_boost) * agent.role * (0.75 + 0.25 * scarcity)
                    + 0.04 * success_rate,
                    0.0,
                    1.0,
                )
                if rng.random() < config.memory_refresh_rate:
                    refresh = clip_position(
                        [
                            value + rng.gauss(0.0, 0.12 * span * (1.0 + scarcity))
                            for value, span in zip(best_position, spans)
                        ],
                        problem.bounds,
                    )
                    refresh_fitness = evaluate_dynamic(problem.name, refresh, shift)
                    if refresh_fitness < agent.fitness:
                        agent.position = list(refresh)
                        agent.fitness = refresh_fitness
                        agent.best_position = list(refresh)
                        agent.best_fitness = refresh_fitness
            scout_count = max(1, int(round(config.scout_injection_ratio * static_config.initial_agents)))
            for _ in range(scout_count):
                if len(population) >= static_config.maximum_agents:
                    break
                scout = init_agent(
                    lambda position: evaluate_dynamic(problem.name, position, shift),
                    problem.bounds,
                    static_config,
                    rng,
                    anchor=None,
                    spread=[0.18 * span for span in spans],
                    role=0.12 + 0.10 * rng.random(),
                )
                scout.fitness = evaluate_dynamic(problem.name, scout.position, shift)
                scout.best_position = list(scout.position)
                scout.best_fitness = scout.fitness
                population.append(scout)
            reevaluate_population(shift)

        center = mean_position(population)
        spread = [
            max(
                0.02 * span,
                math.sqrt(sum((agent.position[idx] - center[idx]) ** 2 for agent in population) / max(1, len(population))),
            )
            for idx, span in enumerate(spans)
        ]
        progress_scale_value = median_upper_scale(
            recent_gains,
            fallback=max(1.0, abs(best_fitness) * 0.1 + 1.0),
        )
        reserve_target = clamp(
            (sum(agent.energy for agent in population) + resource_pool) / max(1, len(population) + 1),
            1.0,
            0.6 * static_config.max_energy,
        )
        crowding = len(population) / max(1, static_config.maximum_agents)
        improvements_this_round = 0
        recovery_pressure = clamp(last_change_gap / max(1.0, progress_scale_value), 0.0, 1.0)
        response_factor = response_rounds_remaining / max(1, config.recovery_window)
        evaluations: list[tuple[object, float, list[float], float]] = []

        for agent in population:
            local_best_position = list(agent.position)
            local_best_fitness = agent.fitness
            step_scale = 0.08 + 0.22 * (1.0 - agent.role) + 0.18 * recovery_pressure + 0.16 * response_factor
            for _ in range(3):
                direction = random_unit_vector(rng, dimensions)
                candidate = clip_position(
                    [value + direction_delta * width * step_scale for value, direction_delta, width in zip(agent.position, direction, spread)],
                    problem.bounds,
                )
                candidate_fitness = evaluate_dynamic(problem.name, candidate, shift)
                if candidate_fitness < local_best_fitness:
                    local_best_position = candidate
                    local_best_fitness = candidate_fitness

            exploit_candidate = clip_position(
                [
                    value
                    + (0.22 + 0.30 * agent.role + 0.18 * config.reinvestment_bias) * (personal - value)
                    + (0.16 + 0.22 * agent.role) * (global_best - value)
                    + rng.gauss(0.0, width * (0.06 + 0.10 * recovery_pressure))
                    for value, personal, global_best, width in zip(agent.position, agent.best_position, best_position, spread)
                ],
                problem.bounds,
            )
            exploit_fitness = evaluate_dynamic(problem.name, exploit_candidate, shift)
            if exploit_fitness < local_best_fitness:
                local_best_position = exploit_candidate
                local_best_fitness = exploit_fitness

            improvement = max(0.0, agent.fitness - local_best_fitness)
            if improvement > 0.0:
                recent_gains.append(improvement)
                improvements_this_round += 1
            if len(recent_gains) > 24:
                recent_gains = recent_gains[-24:]

            normalized_gain = improvement / max(progress_scale_value, 1e-9)
            gross_reward = min(8.0, math.log1p(max(0.0, normalized_gain)))
            maintenance = (
                0.24 + 0.20 * agent.role + 0.18 * crowding + 0.16 * (1.0 - success_rate) + 0.22 * recovery_pressure + 0.10 * response_factor
            )
            communal_share = clamp(
                0.28 + 0.18 * (1.0 - agent.role) + 0.14 * (1.0 - success_rate) + 0.10 * recovery_pressure,
                0.22,
                0.76,
            )
            private_reward = gross_reward * (1.0 - communal_share)
            resource_pool += maintenance + gross_reward * communal_share

            agent.position = list(local_best_position)
            agent.fitness = local_best_fitness
            agent.energy = clamp(agent.energy + private_reward - maintenance, 0.0, static_config.max_energy)
            agent.last_gain = normalized_gain
            agent.age += 1
            agent.stagnation = 0 if improvement > 0.0 else agent.stagnation + 1
            if agent.fitness < agent.best_fitness:
                agent.best_fitness = agent.fitness
                agent.best_position = list(agent.position)

            role_signal = math.tanh(
                0.80 * normalized_gain
                - 0.40 * recovery_pressure
                - 0.40 * (1.0 - success_rate)
                + 0.15 * math.tanh(agent.energy / max(reserve_target, 1e-9) - 1.0)
            )
            agent.role = clamp(agent.role + static_config.role_feedback_rate * role_signal, 0.0, 1.0)

            if agent.fitness < best_fitness:
                best_fitness = agent.fitness
                best_position = list(agent.position)

            evaluations.append((agent, improvement, list(spread), reserve_target))

        success_rate = (
            (1.0 - config.success_feedback_rate) * success_rate
            + config.success_feedback_rate * (improvements_this_round / max(1, len(population)))
        )

        survivors = []
        spawned = []
        for agent, improvement, local_spread, reserve_target in evaluations:
            death_floor = 0.24 * reserve_target * (1.0 + 0.20 * crowding + 0.35 * recovery_pressure)
            if agent.energy < death_floor and agent.stagnation > 1:
                resource_pool += 0.65 * agent.energy
                continue

            survivors.append(agent)
            surplus = agent.energy - reserve_target
            free_slots = static_config.maximum_agents - (len(survivors) + len(spawned))
            if improvement > 0.0 and surplus > 0.10 * reserve_target and resource_pool > 0.30 * reserve_target and free_slots > 0:
                subsidy = min((0.26 + 0.14 * config.reinvestment_bias) * reserve_target, resource_pool)
                parent_share = min(0.22 * reserve_target, 0.22 * agent.energy)
                child_energy = subsidy + parent_share
                resource_pool -= subsidy
                agent.energy = max(0.0, agent.energy - parent_share)
                child = init_agent(
                    lambda position: evaluate_dynamic(problem.name, position, shift),
                    problem.bounds,
                    static_config,
                    rng,
                    anchor=agent.best_position,
                    spread=[max(0.01 * span, width * (0.55 + 0.20 * recovery_pressure)) for width, span in zip(local_spread, spans)],
                    role=clamp(0.18 + 0.54 * agent.role + 0.16 * success_rate, 0.0, 1.0),
                )
                child.fitness = evaluate_dynamic(problem.name, child.position, shift)
                child.best_position = list(child.position)
                child.best_fitness = child.fitness
                child.energy = min(static_config.max_energy, child_energy)
                spawned.append(child)
                if child.fitness < best_fitness:
                    best_fitness = child.fitness
                    best_position = list(child.position)

        population = survivors + spawned

        while len(population) < static_config.minimum_agents:
            anchor = best_position if rng.random() < clamp(0.32 + 0.30 * success_rate, 0.20, 0.78) else None
            scout = init_agent(
                lambda position: evaluate_dynamic(problem.name, position, shift),
                problem.bounds,
                static_config,
                rng,
                anchor=anchor,
                spread=[max(0.02 * span, width * 0.95) for width, span in zip(spread, spans)],
                role=clamp(0.10 + 0.24 * success_rate - 0.10 * recovery_pressure, 0.0, 0.70),
            )
            scout.fitness = evaluate_dynamic(problem.name, scout.position, shift)
            scout.best_position = list(scout.position)
            scout.best_fitness = scout.fitness
            draw = min(resource_pool, 0.8 * reserve_target)
            resource_pool -= draw
            scout.energy = min(static_config.max_energy, max(0.8, 0.55 * reserve_target + draw))
            population.append(scout)
            if scout.fitness < best_fitness:
                best_fitness = scout.fitness
                best_position = list(scout.position)

        resource_pool *= clamp(0.985 - 0.050 * crowding - 0.020 * success_rate, 0.88, 0.99)
        resource_pool = max(resource_pool, 0.0)
        last_change_gap *= 0.88
        response_rounds_remaining = max(0, response_rounds_remaining - 1)
        offline_error = best_fitness - problem.optimum_value
        offline_errors.append(offline_error)
        history.append(
            {
                "iteration": iteration,
                "best_fitness": round(best_fitness, 6),
                "offline_error": round(offline_error, 6),
                "population": len(population),
                "resource_pool": round(resource_pool, 6),
                "success_rate": round(success_rate, 6),
                "mean_role": round(sum(agent.role for agent in population) / max(1, len(population)), 6),
                "regime_id": regime_id,
                "change_gap": round(last_change_gap, 6),
            }
        )

    return {
        "problem": problem.name,
        "best_fitness": best_fitness,
        "best_position": best_position,
        "offline_error": sum(offline_errors) / max(1, len(offline_errors)),
        "history": history,
    }
