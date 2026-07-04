from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ..common import clamp, median_upper_scale


@dataclass(frozen=True)
class TSPProblem:
    name: str
    distance_matrix: list[list[float]]
    coordinates: list[tuple[float, float]] | None = None
    known_optimum: float | None = None

    @property
    def city_count(self) -> int:
        return len(self.distance_matrix)

    def route_length(self, route: list[int]) -> float:
        total = 0.0
        for index, city in enumerate(route):
            nxt = route[(index + 1) % len(route)]
            total += self.distance_matrix[city][nxt]
        return total


@dataclass
class MMAOTSPConfig:
    iterations: int = 180
    seed: int = 11
    initial_agents: int = 8
    minimum_agents: int = 4
    maximum_agents: int = 18
    initial_energy: float = 6.0
    max_energy: float = 16.0
    initial_pool: float = 2.0
    progress_window: int = 24
    communal_share: float = 0.25
    maintenance_cost: float = 0.14
    local_search_trials: int = 16
    kick_strength: int = 2
    role_feedback_rate: float = 0.12
    branch_children: int = 1
    branch_pool_threshold: float = 2.4
    branch_energy_threshold: float = 4.0
    respawn_floor: int = 4
    nearest_neighbor_probability: float = 0.55


@dataclass
class TSPAgent:
    route: list[int]
    energy: float
    role: float
    distance: float
    best_route: list[int]
    best_distance: float
    stagnation: int = 0
    last_gain: float = 0.0


def undirected_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def random_route(city_count: int, rng: random.Random) -> list[int]:
    route = list(range(city_count))
    rng.shuffle(route)
    return route


def nearest_neighbor_route(problem: TSPProblem, rng: random.Random, start: int | None = None) -> list[int]:
    city_count = problem.city_count
    current = rng.randrange(city_count) if start is None else start
    route = [current]
    unvisited = set(range(city_count))
    unvisited.remove(current)
    while unvisited:
        current = min(unvisited, key=lambda city: problem.distance_matrix[route[-1]][city])
        route.append(current)
        unvisited.remove(current)
    return route


def two_opt(route: list[int], i: int, j: int) -> list[int]:
    return route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]


def sampled_two_opt(
    route: list[int],
    problem: TSPProblem,
    rng: random.Random,
    trials: int,
) -> tuple[list[int], float]:
    best_route = route[:]
    best_distance = problem.route_length(best_route)
    city_count = len(route)
    for _ in range(max(1, trials)):
        i = rng.randrange(0, city_count - 2)
        j = rng.randrange(i + 2, city_count)
        if i == 0 and j == city_count - 1:
            continue
        candidate = two_opt(best_route, i, j)
        candidate_distance = problem.route_length(candidate)
        if candidate_distance + 1e-12 < best_distance:
            best_route = candidate
            best_distance = candidate_distance
    return best_route, best_distance


def mutate_route(route: list[int], rng: random.Random, strength: int) -> list[int]:
    candidate = route[:]
    city_count = len(candidate)
    for _ in range(max(1, strength)):
        i, j = sorted(rng.sample(range(city_count), 2))
        candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def edge_memory_from_agents(agents: list[TSPAgent]) -> dict[tuple[int, int], float]:
    best_distance = min(agent.best_distance for agent in agents)
    weights: dict[tuple[int, int], float] = {}
    max_energy = max(a.energy for a in agents)
    for agent in agents:
        quality = 1.0 / max(1e-9, agent.best_distance - best_distance + 1.0)
        energy_weight = 0.5 + agent.energy / max(1.0, max_energy)
        score = quality * energy_weight
        route = agent.best_route
        for index, city in enumerate(route):
            edge = undirected_edge(city, route[(index + 1) % len(route)])
            weights[edge] = weights.get(edge, 0.0) + score
    return weights


def guided_route(problem: TSPProblem, edge_weights: dict[tuple[int, int], float], rng: random.Random) -> list[int]:
    city_count = problem.city_count
    max_distance = max(max(row) for row in problem.distance_matrix)
    current = rng.randrange(city_count)
    route = [current]
    unvisited = set(range(city_count))
    unvisited.remove(current)
    while unvisited:
        next_city = max(
            unvisited,
            key=lambda city: (
                edge_weights.get(undirected_edge(route[-1], city), 0.0)
                - problem.distance_matrix[route[-1]][city] / max(max_distance, 1e-9)
                + rng.random() * 1e-3
            ),
        )
        route.append(next_city)
        unvisited.remove(next_city)
    return route


def make_agent(
    problem: TSPProblem,
    rng: random.Random,
    config: MMAOTSPConfig,
    route: list[int] | None = None,
    role: float | None = None,
    energy: float | None = None,
) -> TSPAgent:
    if route is None:
        if rng.random() < config.nearest_neighbor_probability:
            route = nearest_neighbor_route(problem, rng)
        else:
            route = random_route(problem.city_count, rng)
    distance = problem.route_length(route)
    actual_role = rng.uniform(0.2, 0.8) if role is None else role
    actual_energy = config.initial_energy if energy is None else energy
    return TSPAgent(
        route=route[:],
        energy=actual_energy,
        role=clamp(actual_role, 0.0, 1.0),
        distance=distance,
        best_route=route[:],
        best_distance=distance,
    )


def optimize_tsp_problem(problem: TSPProblem, config: MMAOTSPConfig) -> dict[str, object]:
    rng = random.Random(config.seed)
    agents = [
        make_agent(problem, rng, config)
        for _ in range(max(config.initial_agents, config.minimum_agents))
    ]
    resource_pool = max(0.0, config.initial_pool)
    recent_gains: list[float] = []
    history: list[dict[str, float | int]] = []

    def record(iteration: int) -> None:
        best_agent = min(agents, key=lambda agent: agent.best_distance)
        history.append(
            {
                "iteration": iteration,
                "best_distance": float(best_agent.best_distance),
                "resource_pool": float(resource_pool),
                "success_rate": float(sum(1 for agent in agents if agent.last_gain > 0.0) / len(agents)),
                "mean_role": float(sum(agent.role for agent in agents) / len(agents)),
                "population": int(len(agents)),
                "total_private_energy": float(sum(agent.energy for agent in agents)),
            }
        )

    record(0)

    for iteration in range(1, config.iterations + 1):
        edge_weights = edge_memory_from_agents(agents)
        fallback_scale = min(agent.best_distance for agent in agents) * 0.02
        scale = median_upper_scale(recent_gains[-config.progress_window :], fallback_scale)

        for agent in agents:
            current_route = agent.route[:]
            current_distance = agent.distance

            exploit_trials = max(3, int(config.local_search_trials * (0.45 + 0.75 * agent.role)))
            explore_trials = max(2, int(config.local_search_trials * (0.35 + 0.55 * (1.0 - agent.role))))

            local_route, local_distance = sampled_two_opt(current_route, problem, rng, exploit_trials)

            kicked_route = mutate_route(
                current_route,
                rng,
                strength=max(1, int(config.kick_strength * (0.8 + 1.5 * (1.0 - agent.role)))),
            )
            kicked_route, kicked_distance = sampled_two_opt(kicked_route, problem, rng, explore_trials)

            guided_candidate = guided_route(problem, edge_weights, rng)
            guided_candidate, guided_distance = sampled_two_opt(
                guided_candidate,
                problem,
                rng,
                max(3, config.local_search_trials // 2),
            )

            accepted_route = current_route
            accepted_distance = current_distance
            for candidate_route, candidate_distance in (
                (local_route, local_distance),
                (kicked_route, kicked_distance),
                (guided_candidate, guided_distance),
            ):
                if candidate_distance + 1e-12 < accepted_distance:
                    accepted_route = candidate_route
                    accepted_distance = candidate_distance

            raw_gain = max(0.0, current_distance - accepted_distance)
            normalized_gain = raw_gain / scale
            reward = min(4.0, math.log1p(normalized_gain))
            resource_pool += config.communal_share * reward

            maintenance = config.maintenance_cost * (0.9 + 0.35 * agent.role + 0.1 * min(6, agent.stagnation))
            agent.energy = clamp(
                agent.energy + (1.0 - config.communal_share) * reward - maintenance,
                0.0,
                config.max_energy,
            )
            agent.last_gain = raw_gain

            if accepted_distance + 1e-12 < current_distance:
                agent.route = accepted_route[:]
                agent.distance = accepted_distance
                if accepted_distance + 1e-12 < agent.best_distance:
                    agent.best_distance = accepted_distance
                    agent.best_route = accepted_route[:]
                agent.stagnation = 0
                agent.role = clamp(
                    agent.role + config.role_feedback_rate * (0.25 + 0.5 * min(1.0, normalized_gain)),
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

            recent_gains.append(raw_gain)

        removable = [agent for agent in agents if agent.energy <= 0.05]
        for agent in removable:
            if len(agents) <= config.minimum_agents:
                break
            resource_pool += 0.15 * agent.energy
            agents.remove(agent)

        branch_candidates = sorted(
            agents,
            key=lambda agent: (agent.last_gain, -agent.best_distance),
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
                child_route = mutate_route(parent.best_route, rng, strength=1)
                child_route, _ = sampled_two_opt(
                    child_route,
                    problem,
                    rng,
                    max(3, config.local_search_trials // 3),
                )
                child_energy = min(config.initial_energy * 0.75, resource_pool)
                child = make_agent(
                    problem,
                    rng,
                    config,
                    route=child_route,
                    role=0.5 * parent.role + 0.2,
                    energy=child_energy,
                )
                agents.append(child)
                resource_pool = max(0.0, resource_pool - 0.6 * child_energy)

        while len(agents) < config.minimum_agents or (
            len(agents) < min(config.respawn_floor, config.maximum_agents) and resource_pool > 1.2
        ):
            if len(agents) >= config.maximum_agents:
                break
            spawn_energy = min(config.initial_energy, max(1.0, resource_pool))
            seed_route = guided_route(problem, edge_weights, rng) if rng.random() < 0.7 else None
            child = make_agent(
                problem,
                rng,
                config,
                route=seed_route,
                role=0.25 if seed_route is None else 0.35,
                energy=spawn_energy,
            )
            agents.append(child)
            resource_pool = max(0.0, resource_pool - 0.5 * spawn_energy)

        record(iteration)

    best_agent = min(agents, key=lambda agent: agent.best_distance)
    return {
        "problem": problem.name,
        "best_route": best_agent.best_route[:],
        "best_distance": float(best_agent.best_distance),
        "history": history,
    }
