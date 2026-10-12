"""Equal-budget terminal policy-gradient, random and CEM parameter search.

Only actual exported tokenizers receive the shared multi-objective reward.
All methods use the same initial policy, tasks, proposal scale, candidate
features and training-only incumbent selection. No validation/test selection.
Parameter score-function learning avoids summing hundreds of action scores.
"""
from dataclasses import asdict, dataclass
import time
import numpy as np
from .rl_transfer_v7 import TransferConfig, rollout as reference_rollout, task_rows
from .corpus_index_v8 import CorpusFitIndex
from .rl_route_v6 import ReferenceReward
from .trajectory_v5 import TrajectoryConfig, rloo_gradient


@dataclass(frozen=True)
class SearchConfig:
    rounds: int = 4
    samples: int = 4
    sigma: float = .35
    learning_rate: float = .12
    fit_limit: int = 1024
    reward_limit: int = 64
    training_capacity: int = 768
    candidates: int = 16


def reward_config():
    return TrajectoryConfig(code_weight=.04, command_weight=.15, number_weight=.03,
                            tree_weight=.5, robustness_weight=.04, rare_weight=.02)


def rollout(*args,**kwargs):
    return reference_rollout(*args,**kwargs,index_class=CorpusFitIndex)


def gaussian_score(theta, mean, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('positive finite proposal scale required')
    return (np.asarray(theta) - np.asarray(mean)) / sigma**2


def optimize(fit, reward, atoms, config, seed, algorithm):
    if algorithm not in ('parameter_rl', 'random_search', 'cem_search'):
        raise ValueError('unknown terminal optimizer')
    if config.samples < 2 or config.rounds < 1:
        raise ValueError('multiple independent samples and positive rounds required')
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    initial = np.zeros(24); initial[0] = 2.
    mean = initial.copy(); first = np.zeros(24); second = np.zeros(24)
    merge_config = TransferConfig(candidates=config.candidates)
    tasks = []
    for task in (0, 1):
        rows = task_rows(fit, config.fit_limit, task)
        held = task_rows(reward, config.reward_limit, task)
        tasks.append((rows, ReferenceReward(rows, held, reward_config()), held))
    evaluations = []; history = []

    def evaluate(theta, iteration, sample):
        values = []
        for task, (rows, objective, held) in enumerate(tasks):
            model, _, _ = rollout(rows, atoms, config.training_capacity, theta, merge_config)
            stats = objective.score(model)
            values.append(stats)
        result = {'iteration': iteration, 'sample': sample, 'policy': theta.tolist(),
                  'task_statistics': values, 'utility': float(np.mean([v['utility'] for v in values]))}
        evaluations.append(result)
        return result

    evaluate(initial, -1, 0)
    for iteration in range(config.rounds):
        before = mean.copy()
        proposals = [before + config.sigma*rng.normal(size=24) for _ in range(config.samples)]
        batch = [evaluate(theta, iteration, sample) for sample, theta in enumerate(proposals)]
        scores = np.asarray([gaussian_score(theta, before, config.sigma) for theta in proposals])
        gradient, advantages = rloo_gradient(scores, [-b['utility'] for b in batch])
        # Fixed reward scale, rather than an action-dependent fitted baseline.
        scaled = gradient/.05
        scaled /= max(1., np.linalg.norm(scaled))
        if algorithm == 'parameter_rl':
            first = .9*first+.1*scaled; second = .999*second+.001*scaled**2
            mean += config.learning_rate*(first/(1-.9**(iteration+1)))/(np.sqrt(second/(1-.999**(iteration+1)))+1e-8)
        elif algorithm == 'cem_search':
            elites = sorted(batch, key=lambda b: (b['utility'], b['sample']))[:config.samples//2]
            mean = .5*before+.5*np.mean([b['policy'] for b in elites], axis=0)
        history.append({'iteration': iteration, 'mean_before': before.tolist(), 'mean_after': mean.tolist(),
                        'gradient': gradient.tolist(), 'advantages': advantages.tolist(),
                        'update': algorithm, 'terminal_evaluations_so_far': len(evaluations)*len(tasks)})
    incumbent = min(evaluations, key=lambda e: (e['utility'], e['iteration'], e['sample']))
    elapsed = time.perf_counter()-started
    return {'algorithm': algorithm, 'seed': seed, 'config': asdict(config),
            'fit_index_backend':'CorpusFitIndex-v8',
            'reward_config': asdict(reward_config()), 'initial_policy': initial.tolist(),
            'policy': incumbent['policy'], 'last_distribution_mean': mean.tolist(),
            'incumbent_utility': incumbent['utility'], 'evaluations': evaluations, 'history': history,
            'terminal_evaluations': len(evaluations)*len(tasks), 'candidate_count': len(evaluations),
            'task_fit_ids': [[r['id'] for r in rows] for rows, _, _ in tasks],
            'task_reward_ids': [[r['id'] for r in rows] for _, _, rows in tasks],
            'training_seconds': elapsed, 'validation_or_test_used': False,
            'selection': 'same best scored training-reward incumbent including common initial policy'}


def export(fit, reward, atoms, capacity, policy):
    model, _, _ = rollout(fit, atoms, capacity, np.asarray(policy['policy']),
                           TransferConfig(candidates=policy['config']['candidates']))
    model.seed = policy['seed']; model.reward_ids = [r['id'] for r in reward]
    model.train_seconds += policy['training_seconds']
    model.method = 'v8_'+policy['algorithm']
    model.logs[0].update(optimizer=policy['algorithm'], terminal_evaluations=policy['terminal_evaluations'],
                         reward_config=policy['reward_config'], validation_or_test_used=False)
    return model
