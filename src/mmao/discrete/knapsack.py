from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ..common import clamp, median_upper_scale


@dataclass(frozen=True)
class KnapsackProblem:
    name: str
    profits: list[int]
    weights: list[list[int]]
    capacities: list[int]
    known_optimum: int | None = None

    @property
    def item_count(self) -> int:
        return len(self.profits)

    def feasible(self, solution: list[int]) -> bool:
        return all(load <= cap for load, cap in zip(self.usage(solution), self.capacities))

    def usage(self, solution: list[int]) -> list[int]:
        return [
            sum(weight * bit for weight, bit in zip(row, solution))
            for row in self.weights
        ]

    def profit(self, solution: list[int]) -> int:
        return sum(profit * bit for profit, bit in zip(self.profits, solution))


@dataclass
class MMAOKnapsackConfig:
    iterations: int = 90
    seed: int = 23
    initial_agents: int = 10
    minimum_agents: int = 5
    maximum_agents: int = 18
    initial_energy: float = 8.0
    max_energy: float = 18.0
    initial_pool_ratio: float = 0.88
    progress_window: int = 24
    reward_scale: float = 1.6
    reward_clip: float = 6.0
    success_feedback_rate: float = 0.18
    role_update_rate: float = 0.16
    bitflip_k_min: int = 2
    bitflip_k_base: int = 8
    fixed_role: float | None = None
    use_guided_memory: bool = True


@dataclass
class KnapsackAgent:
    solution: list[int]
    energy: float
    role: float
    age: int
    profit: int
    best_solution: list[int]
    best_profit: int
    stagnation: int
    last_gain: float


def density_order(problem: KnapsackProblem) -> list[int]:
    density: list[tuple[float, int]] = []
    for item, profit in enumerate(problem.profits):
        normalized_weight = sum(
            problem.weights[row][item] / max(1, problem.capacities[row])
            for row in range(len(problem.capacities))
        )
        density.append((profit / max(1e-9, normalized_weight), item))
    density.sort(reverse=True)
    return [item for _, item in density]


def build_density_rank(problem: KnapsackProblem) -> tuple[list[int], dict[int, int]]:
    order = density_order(problem)
    return order, {item: idx for idx, item in enumerate(order)}


def repair_solution(
    solution: list[int],
    problem: KnapsackProblem,
    order: list[int],
    rank: dict[int, int],
) -> list[int]:
    repaired = list(solution)
    usage = problem.usage(repaired)

    def feasible_usage(loads: list[int]) -> bool:
        return all(load <= capacity for load, capacity in zip(loads, problem.capacities))

    while not feasible_usage(usage):
        chosen = [idx for idx, bit in enumerate(repaired) if bit]
        if not chosen:
            break
        worst_item = max(
            chosen,
            key=lambda idx: (
                sum(problem.weights[row][idx] / max(1, problem.capacities[row]) for row in range(len(problem.capacities)))
                / max(1, problem.profits[idx]),
                rank[idx],
            ),
        )
        repaired[worst_item] = 0
        for row in range(len(problem.capacities)):
            usage[row] -= problem.weights[row][worst_item]

    for item in order:
        if repaired[item]:
            continue
        next_usage = [usage[row] + problem.weights[row][item] for row in range(len(problem.capacities))]
        if feasible_usage(next_usage):
            repaired[item] = 1
            usage = next_usage
    return repaired


def random_feasible_solution(
    rng: random.Random,
    problem: KnapsackProblem,
    order: list[int],
    rank: dict[int, int],
) -> list[int]:
    solution = [0] * problem.item_count
    ordering = list(range(problem.item_count))
    rng.shuffle(ordering)
    for item in ordering:
        if rng.random() < 0.5:
            solution[item] = 1
            if not problem.feasible(solution):
                solution[item] = 0
    return repair_solution(solution, problem, order, rank)


def greedy_density_solution(problem: KnapsackProblem, order: list[int]) -> list[int]:
    solution = [0] * problem.item_count
    for item in order:
        solution[item] = 1
        if not problem.feasible(solution):
            solution[item] = 0
    return solution


def local_improve(
    solution: list[int],
    problem: KnapsackProblem,
    rng: random.Random,
    attempts: int,
    order: list[int],
    rank: dict[int, int],
    memory_scores: list[float] | None = None,
) -> tuple[list[int], int]:
    best_solution = list(solution)
    best_profit = problem.profit(best_solution)
    for _ in range(max(1, attempts)):
        candidate = list(best_solution)
        pivot = (
            rng.randrange(problem.item_count)
            if memory_scores is None
            else max(range(problem.item_count), key=lambda idx: memory_scores[idx] + 0.05 * rng.random())
        )
        candidate[pivot] = 1 - candidate[pivot]
        if rng.random() < 0.35:
            second = rng.randrange(problem.item_count)
            if second != pivot:
                candidate[second] = 1 - candidate[second]
        candidate = repair_solution(candidate, problem, order, rank)
        candidate_profit = problem.profit(candidate)
        if candidate_profit > best_profit:
            best_solution = candidate
            best_profit = candidate_profit
    return best_solution, best_profit


def build_item_memory(population: list[KnapsackAgent], item_count: int) -> list[float]:
    scores = [0.0] * item_count
    if not population:
        return scores
    mean_profit = sum(agent.profit for agent in population) / len(population)
    for agent in population:
        weight = max(agent.energy, 1e-6) * (1.0 + 0.30 * agent.role)
        quality = clamp(agent.profit / max(1.0, mean_profit), 0.4, 2.0)
        for item, bit in enumerate(agent.solution):
            if bit:
                scores[item] += weight * quality
    return scores


def init_agent(
    problem: KnapsackProblem,
    config: MMAOKnapsackConfig,
    rng: random.Random,
    order: list[int],
    rank: dict[int, int],
    anchor: list[int] | None = None,
    role: float | None = None,
) -> KnapsackAgent:
    if anchor is None:
        solution = random_feasible_solution(rng, problem, order, rank)
    else:
        solution = list(anchor)
        flip_count = max(1, int(round(1 + (1.0 - (role if role is not None else 0.5)) * 4)))
        for _ in range(flip_count):
            idx = rng.randrange(len(solution))
            solution[idx] = 1 - solution[idx]
        solution = repair_solution(solution, problem, order, rank)
    profit = problem.profit(solution)
    initial_role = clamp(
        config.fixed_role if config.fixed_role is not None else (role if role is not None else rng.uniform(0.18, 0.82)),
        0.0,
        1.0,
    )
    return KnapsackAgent(
        solution=solution,
        energy=config.initial_energy,
        role=initial_role,
        age=0,
        profit=profit,
        best_solution=list(solution),
        best_profit=profit,
        stagnation=0,
        last_gain=0.0,
    )


def optimize_knapsack_problem(problem: KnapsackProblem, config: MMAOKnapsackConfig) -> dict[str, object]:
    rng = random.Random(config.seed)
    order, rank = build_density_rank(problem)
    population = [init_agent(problem, config, rng, order, rank) for _ in range(config.initial_agents)]
    best_agent = max(population, key=lambda agent: agent.profit)
    best_solution = list(best_agent.solution)
    best_profit = best_agent.profit
    resource_pool = config.initial_pool_ratio * config.initial_agents * config.initial_energy
    success_rate = 0.45
    recent_gains: list[float] = []

    history: list[dict[str, float | int]] = [
        {
            "iteration": 0,
            "best_profit": best_profit,
            "population": len(population),
            "total_energy": round(sum(agent.energy for agent in population), 6),
            "resource_pool": round(resource_pool, 6),
            "success_rate": round(success_rate, 6),
            "mean_role": round(sum(agent.role for agent in population) / len(population), 6),
        }
    ]

    for iteration in range(1, config.iterations + 1):
        item_memory = build_item_memory(population, problem.item_count) if config.use_guided_memory else None
        reserve_target = clamp(
            (sum(agent.energy for agent in population) + resource_pool) / max(1, len(population) + 1),
            1.0,
            0.60 * config.max_energy,
        )
        progress_scale_value = median_upper_scale(recent_gains, fallback=max(1.0, 0.05 * max(1, best_profit)))
        crowding = len(population) / max(1, config.maximum_agents)
        improvements_this_round = 0
        evaluations: list[tuple[KnapsackAgent, float, float]] = []

        for agent in population:
            old_profit = agent.profit
            structural_scale = clamp(
                0.18 + 0.32 * (1.0 - agent.role) + 0.22 * (1.0 - success_rate) + 0.10 * min(1.0, agent.stagnation / 4.0),
                0.10,
                1.0,
            )
            attempts = max(config.bitflip_k_min, int(round(config.bitflip_k_min + config.bitflip_k_base * structural_scale)))

            anchors = [agent.best_solution]
            if agent.best_profit < best_profit:
                anchors.append(best_solution)
            if item_memory is not None and rng.random() < 0.55:
                anchors.append(greedy_density_solution(problem, order))

            candidates = [(list(agent.solution), agent.profit)]
            for anchor in anchors:
                candidate, candidate_profit = local_improve(anchor, problem, rng, attempts, order, rank, item_memory)
                candidates.append((candidate, candidate_profit))

            candidate_solution, candidate_profit = max(candidates, key=lambda item: item[1])
            improvement = max(0.0, candidate_profit - old_profit)
            if improvement > 0.0:
                improvements_this_round += 1
                recent_gains.append(float(improvement))
            if len(recent_gains) > config.progress_window:
                recent_gains = recent_gains[-config.progress_window :]

            normalized_gain = improvement / max(progress_scale_value, 1e-9)
            gross_reward = min(config.reward_clip, config.reward_scale * math.log1p(normalized_gain))
            maintenance = 0.20 + 0.22 * agent.role + 0.16 * crowding + 0.18 * (1.0 - success_rate)
            communal_share = clamp(0.28 + 0.18 * (1.0 - agent.role) + 0.15 * (1.0 - success_rate), 0.22, 0.70)
            private_reward = gross_reward * (1.0 - communal_share)
            resource_pool += maintenance + gross_reward * communal_share

            if candidate_profit >= old_profit:
                agent.solution = candidate_solution
                agent.profit = candidate_profit
                agent.stagnation = 0 if improvement > 0.0 else agent.stagnation + 1
            else:
                agent.stagnation += 1

            agent.age += 1
            agent.energy = clamp(agent.energy + private_reward - maintenance, 0.0, config.max_energy)
            agent.last_gain = normalized_gain

            role_signal = math.tanh(
                0.85 * normalized_gain
                - 0.50 * (1.0 - success_rate)
                - 0.12 * min(agent.stagnation, 4)
                + 0.18 * math.tanh(agent.energy / max(reserve_target, 1e-9) - 1.0)
            )
            if config.fixed_role is not None:
                agent.role = clamp(config.fixed_role, 0.0, 1.0)
            else:
                agent.role = clamp(agent.role + config.role_update_rate * role_signal, 0.0, 1.0)

            if agent.profit > agent.best_profit:
                agent.best_profit = agent.profit
                agent.best_solution = list(agent.solution)
                agent.stagnation = 0

            if agent.profit > best_profit:
                best_profit = agent.profit
                best_solution = list(agent.solution)

            evaluations.append((agent, improvement, reserve_target))

        success_rate = (
            (1.0 - config.success_feedback_rate) * success_rate
            + config.success_feedback_rate * (improvements_this_round / max(1, len(population)))
        )

        survivors: list[KnapsackAgent] = []
        spawned: list[KnapsackAgent] = []
        for agent, improvement, reserve_target in evaluations:
            death_floor = 0.22 * reserve_target * (1.0 + 0.22 * crowding)
            if agent.energy < death_floor and agent.stagnation > 1:
                resource_pool += 0.65 * agent.energy
                continue
            survivors.append(agent)

            surplus = agent.energy - reserve_target
            if improvement <= 0.0 or surplus <= 0.10 * reserve_target:
                continue
            if len(survivors) + len(spawned) >= config.maximum_agents:
                continue

            subsidy = min(0.30 * reserve_target, resource_pool)
            parent_share = min(0.24 * reserve_target, 0.22 * agent.energy)
            child_energy = subsidy + parent_share
            if child_energy < 0.55:
                continue
            resource_pool -= subsidy
            agent.energy = max(0.0, agent.energy - parent_share)
            child_role = clamp(
                config.fixed_role if config.fixed_role is not None else 0.18 + 0.50 * agent.role + 0.16 * success_rate,
                0.0,
                1.0,
            )
            child = init_agent(problem, config, rng, order, rank, anchor=agent.best_solution, role=child_role)
            child.energy = min(config.max_energy, child_energy)
            if child.profit > best_profit:
                best_profit = child.profit
                best_solution = list(child.solution)
            spawned.append(child)

        population = survivors + spawned

        while len(population) < config.minimum_agents:
            anchor = best_solution if rng.random() < (0.35 + 0.35 * success_rate) else None
            scout_role = clamp(
                config.fixed_role if config.fixed_role is not None else 0.15 + 0.25 * success_rate,
                0.0,
                1.0,
            )
            scout = init_agent(problem, config, rng, order, rank, anchor=anchor, role=scout_role)
            population.append(scout)
            if scout.profit > best_profit:
                best_profit = scout.profit
                best_solution = list(scout.solution)

        resource_pool *= clamp(0.985 - 0.055 * crowding - 0.025 * success_rate, 0.88, 0.985)
        resource_pool = max(resource_pool, 0.0)

        history.append(
            {
                "iteration": iteration,
                "best_profit": best_profit,
                "population": len(population),
                "total_energy": round(sum(agent.energy for agent in population), 6),
                "resource_pool": round(resource_pool, 6),
                "success_rate": round(success_rate, 6),
                "mean_role": round(sum(agent.role for agent in population) / max(1, len(population)), 6),
            }
        )

    return {
        "problem": problem.name,
        "best_profit": best_profit,
        "best_solution": best_solution,
        "history": history,
    }
