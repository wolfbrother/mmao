from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ..common import clamp, median_upper_scale
from .benchmarks import ContinuousProblem


Vector = list[float]


@dataclass
class MMAOContinuousConfig:
    iterations: int = 160
    seed: int = 7
    initial_agents: int = 8
    minimum_agents: int = 4
    maximum_agents: int = 20
    initial_energy: float = 6.0
    max_energy: float = 16.0
    initial_pool: float = 2.0
    progress_window: int = 24
    communal_share: float = 0.25
    maintenance_cost: float = 0.14
    sense_radius_ratio: float = 0.16
    local_step_ratio: float = 0.11
    global_pull: float = 0.36
    role_feedback_rate: float = 0.12
    branch_children: int = 1
    branch_energy_threshold: float = 4.2
    branch_pool_threshold: float = 2.6
    respawn_floor: int = 4


@dataclass
class ContinuousAgent:
    position: Vector
    energy: float
    role: float
    fitness: float
    best_position: Vector
    best_fitness: float
    age: int = 0
    stagnation: int = 0
    last_gain: float = 0.0


def random_unit_vector(rng: random.Random, dimensions: int) -> Vector:
    values = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        return [1.0] + [0.0] * (dimensions - 1)
    return [value / norm for value in values]


def clip_position(position: Vector, bounds: list[tuple[float, float]]) -> Vector:
    return [min(upper, max(lower, value)) for value, (lower, upper) in zip(position, bounds)]


def mean_position(agents: list[ContinuousAgent]) -> Vector:
    dimensions = len(agents[0].position)
    return [
        sum(agent.position[dimension] for agent in agents) / len(agents)
        for dimension in range(dimensions)
    ]


def squared_distance(a: Vector, b: Vector) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def make_agent(
    rng: random.Random,
    problem: ContinuousProblem,
    energy: float,
    role: float | None = None,
    anchor: Vector | None = None,
    radius: float = 0.0,
) -> ContinuousAgent:
    if anchor is None:
        position = [rng.uniform(lower, upper) for lower, upper in problem.bounds]
    else:
        position = []
        for dimension, (lower, upper) in enumerate(problem.bounds):
            value = anchor[dimension] + rng.gauss(0.0, radius)
            position.append(min(upper, max(lower, value)))
    fitness = problem.evaluate(position)
    actual_role = rng.uniform(0.2, 0.8) if role is None else role
    return ContinuousAgent(
        position=position[:],
        energy=energy,
        role=clamp(actual_role, 0.0, 1.0),
        fitness=fitness,
        best_position=position[:],
        best_fitness=fitness,
    )


def init_agent(
    objective,
    bounds: list[tuple[float, float]],
    config: MMAOContinuousConfig,
    rng: random.Random,
    anchor: Vector | None = None,
    spread: list[float] | None = None,
    role: float | None = None,
) -> ContinuousAgent:
    if anchor is None:
        position = [rng.uniform(lower, upper) for lower, upper in bounds]
    else:
        local_spread = spread or [0.12 * (upper - lower) for lower, upper in bounds]
        position = clip_position(
            [value + rng.gauss(0.0, width) for value, width in zip(anchor, local_spread)],
            bounds,
        )
    fitness = float(objective(position))
    actual_role = rng.uniform(0.2, 0.8) if role is None else role
    return ContinuousAgent(
        position=position[:],
        energy=config.initial_energy,
        role=clamp(actual_role, 0.0, 1.0),
        fitness=fitness,
        best_position=position[:],
        best_fitness=fitness,
    )


def optimize_continuous_problem(
    problem: ContinuousProblem,
    config: MMAOContinuousConfig,
) -> dict[str, object]:
    rng = random.Random(config.seed)
    dimensions = problem.dimension
    domain_scale = math.sqrt(
        sum((upper - lower) ** 2 for lower, upper in problem.bounds) / max(1, dimensions)
    )

    agents = [
        make_agent(rng, problem, config.initial_energy)
        for _ in range(max(config.initial_agents, config.minimum_agents))
    ]
    resource_pool = max(0.0, config.initial_pool)
    recent_gains: list[float] = []
    history: list[dict[str, float | int]] = []
    global_best_agent = min(agents, key=lambda agent: agent.best_fitness)
    global_best_fitness = float(global_best_agent.best_fitness)
    global_best_position = global_best_agent.best_position[:]

    def record(iteration: int) -> None:
        history.append(
            {
                "iteration": iteration,
                "best_fitness": float(global_best_fitness),
                "resource_pool": float(resource_pool),
                "success_rate": float(sum(1 for agent in agents if agent.last_gain > 0.0) / len(agents)),
                "mean_role": float(sum(agent.role for agent in agents) / len(agents)),
                "population": int(len(agents)),
                "total_private_energy": float(sum(agent.energy for agent in agents)),
            }
        )

    record(0)

    for iteration in range(1, config.iterations + 1):
        global_best = min(agents, key=lambda agent: agent.best_fitness)
        center = mean_position(agents)
        mean_spread = math.sqrt(
            sum(squared_distance(agent.position, center) for agent in agents) / max(1, len(agents))
        )
        spread_ratio = min(1.5, mean_spread / max(domain_scale, 1e-9))
        scale = median_upper_scale(recent_gains[-config.progress_window :], domain_scale * 0.02)

        for agent in agents:
            current_position = agent.position[:]
            current_fitness = agent.fitness
            personal_gap = [best - current for best, current in zip(agent.best_position, current_position)]
            global_gap = [best - current for best, current in zip(global_best.best_position, current_position)]

            sense_radius = (
                config.sense_radius_ratio
                * domain_scale
                * (0.55 + 0.65 * spread_ratio)
                * (0.35 + 0.85 * (1.0 - agent.role))
            )
            search_direction = random_unit_vector(rng, dimensions)
            plus = clip_position(
                [value + sense_radius * delta for value, delta in zip(current_position, search_direction)],
                problem.bounds,
            )
            minus = clip_position(
                [value - sense_radius * delta for value, delta in zip(current_position, search_direction)],
                problem.bounds,
            )
            plus_fitness = problem.evaluate(plus)
            minus_fitness = problem.evaluate(minus)
            probe_position = plus if plus_fitness <= minus_fitness else minus
            probe_fitness = min(plus_fitness, minus_fitness)
            probe_gap = [probe - current for probe, current in zip(probe_position, current_position)]

            noise_scale = config.local_step_ratio * domain_scale * (0.35 + 0.75 * (1.0 - agent.role))
            noise_direction = random_unit_vector(rng, dimensions)
            candidate = []
            for dimension in range(dimensions):
                candidate.append(
                    current_position[dimension]
                    + (0.35 + 0.65 * agent.role) * config.global_pull * global_gap[dimension]
                    + 0.25 * agent.role * personal_gap[dimension]
                    + (0.45 + 0.45 * (1.0 - agent.role)) * probe_gap[dimension]
                    + noise_scale * noise_direction[dimension]
                )
            candidate = clip_position(candidate, problem.bounds)
            candidate_fitness = problem.evaluate(candidate)

            accepted_position = current_position
            accepted_fitness = current_fitness
            if probe_fitness < accepted_fitness:
                accepted_position = probe_position
                accepted_fitness = probe_fitness
            if candidate_fitness < accepted_fitness:
                accepted_position = candidate
                accepted_fitness = candidate_fitness

            raw_gain = max(0.0, current_fitness - accepted_fitness)
            normalized_gain = raw_gain / scale
            reward = min(4.0, math.log1p(normalized_gain))
            resource_pool += config.communal_share * reward

            maintenance = config.maintenance_cost * (
                0.9 + 0.4 * agent.role + 0.1 * min(6, agent.stagnation)
            )
            agent.energy = clamp(
                agent.energy + (1.0 - config.communal_share) * reward - maintenance,
                0.0,
                config.max_energy,
            )
            agent.last_gain = raw_gain

            if accepted_fitness < current_fitness - 1e-12:
                agent.position = accepted_position[:]
                agent.fitness = accepted_fitness
                if accepted_fitness < agent.best_fitness:
                    agent.best_fitness = accepted_fitness
                    agent.best_position = accepted_position[:]
                agent.stagnation = 0
                agent.role = clamp(
                    agent.role + config.role_feedback_rate * (0.25 + min(1.0, normalized_gain) * 0.5),
                    0.0,
                    1.0,
                )
            else:
                agent.stagnation += 1
                energy_pressure = 1.0 - agent.energy / max(config.max_energy, 1e-9)
                agent.role = clamp(
                    agent.role - config.role_feedback_rate * (0.3 + 0.45 * energy_pressure),
                    0.0,
                    1.0,
                )

            if agent.fitness < agent.best_fitness:
                agent.best_fitness = agent.fitness
                agent.best_position = agent.position[:]
            if agent.best_fitness < global_best_fitness:
                global_best_fitness = float(agent.best_fitness)
                global_best_position = agent.best_position[:]

            recent_gains.append(raw_gain)

        removable = [agent for agent in agents if agent.energy <= 0.05]
        for agent in removable:
            if len(agents) <= config.minimum_agents:
                break
            resource_pool += 0.15 * agent.energy
            agents.remove(agent)

        branch_candidates = sorted(
            agents,
            key=lambda agent: (agent.last_gain, agent.best_fitness),
            reverse=True,
        )
        for parent in branch_candidates:
            if len(agents) >= config.maximum_agents:
                break
            if resource_pool < config.branch_pool_threshold or parent.energy < config.branch_energy_threshold:
                continue
            for _ in range(config.branch_children):
                if len(agents) >= config.maximum_agents or resource_pool < 0.8:
                    break
                child_energy = min(config.initial_energy * 0.75, resource_pool)
                child_radius = config.local_step_ratio * domain_scale * (0.25 + 0.45 * (1.0 - parent.role))
                child = make_agent(
                    rng,
                    problem,
                    energy=child_energy,
                    role=0.5 * parent.role + 0.2,
                    anchor=parent.best_position,
                    radius=child_radius,
                )
                agents.append(child)
                resource_pool = max(0.0, resource_pool - 0.6 * child_energy)

        while len(agents) < config.minimum_agents or (
            len(agents) < min(config.respawn_floor, config.maximum_agents) and resource_pool > 1.2
        ):
            if len(agents) >= config.maximum_agents:
                break
            best_agent = min(agents, key=lambda agent: agent.best_fitness)
            use_anchor = rng.random() < 0.6
            spawn_energy = min(config.initial_energy, max(1.0, resource_pool))
            child = make_agent(
                rng,
                problem,
                energy=spawn_energy,
                role=0.25 if not use_anchor else 0.35,
                anchor=best_agent.best_position if use_anchor else None,
                radius=config.local_step_ratio * domain_scale * 0.5,
            )
            agents.append(child)
            resource_pool = max(0.0, resource_pool - 0.5 * spawn_energy)

        record(iteration)

    return {
        "problem": problem.name,
        "best_position": global_best_position,
        "best_fitness": float(global_best_fitness),
        "history": history,
    }
