"""Whole-merge-trajectory RL on the v5 environment; official BPE deployment.

All actions in a rollout use one frozen policy. A terminal multiobjective reward
credits the complete merge table, then independent rollouts provide RLOO policy
gradients. This is standalone tokenizer RL, not the uploaded paper's internal LM
gater or its early-exit loss. The online tokenizer never runs this learner.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
import re
import time

import numpy as np

from .measure_v5 import cross_count
from .rl_vocab_v5 import MergeSession, RLVocab, VocabRLConfig, internal_split
from .structure_v3 import structure_fast


@dataclass(frozen=True)
class TrajectoryConfig:
    epochs: int = 4
    samples: int = 4
    learning_rate: float = .25
    temperature: float = 3.
    exploration: float = .05
    gradient_clip: float = 1.
    code_weight: float = .04
    command_weight: float = .15
    number_weight: float = .03
    tree_weight: float = .10
    robustness_weight: float = .04
    rare_weight: float = .02
    baseline: str = 'rloo'
    critic_ridge: float = .1


def probabilities_and_score(features, theta, action=None, temperature=1., exploration=0.):
    logits = features @ theta / temperature
    logits -= logits.max()
    pure = np.exp(logits)
    pure /= pure.sum()
    mixed = (1 - exploration) * pure + exploration / len(pure)
    if action is None:
        return mixed
    score = (1 - exploration) * pure[action] / mixed[action] * (features[action] - pure @ features) / temperature
    return mixed, score


def rloo_gradient(scores, returns):
    """Unnormalized independent-rollout leave-one-out score-function mean."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        raise ValueError("RLOO needs at least two independent trajectories")
    baseline = (returns.sum() - returns) / (len(returns) - 1)
    advantages = returns - baseline
    return advantages @ np.asarray(scores) / len(returns), advantages


def crossfit_critic_gradient(score_steps, state_features, returns, ridge=.1):
    """Learn a state value from OTHER rollouts, never the queried return.

    Conditional on the current prefix, its action is independent of the fitted
    other-rollout baseline. This keeps the score-function expectation intact;
    fitting a critic to its own rollout would not have this guarantee.
    """
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or ridge <= 0:
        raise ValueError('invalid cross-fit critic configuration')
    estimates, mean_advantages, models = [], [], []
    for i, value in enumerate(values):
        other = [j for j in range(len(values)) if j != i]
        x = np.concatenate([state_features[j] for j in other])
        target = np.concatenate([np.full(len(state_features[j]), values[j]) for j in other])
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-6] = 1.
        center[0], scale[0] = 0., 1.  # explicit intercept
        z = (x - center) / scale
        penalty = ridge * len(z) * np.eye(z.shape[1])
        penalty[0, 0] = 0.
        coefficient = np.linalg.solve(z.T @ z + penalty, z.T @ target)
        predictions = ((state_features[i] - center) / scale) @ coefficient
        width = max(.05, float(np.ptp(values[other])))
        predictions = np.clip(predictions, values[other].min() - width, values[other].max() + width)
        advantage = value - predictions
        estimates.append(advantage @ score_steps[i])
        mean_advantages.append(float(advantage.mean()))
        models.append({'excluded_rollout': i, 'trained_on_rollouts': other,
                       'coefficients': coefficient.tolist(),
                       'prediction_mean': float(predictions.mean())})
    estimates = np.asarray(estimates)
    scalar_baseline = (values.sum() - values) / (len(values) - 1)
    reference = np.asarray([(value - scalar_baseline[i]) * np.sum(score_steps[i], axis=0) for i, value in enumerate(values)])
    current_variance = float(np.mean(np.var(estimates, axis=0, ddof=1)))
    reference_variance = float(np.mean(np.var(reference, axis=0, ddof=1)))
    diagnostics = {'cross_fitted_value_heads': models,
                   'paired_gradient_variance': current_variance,
                   'rloo_on_identical_rollouts_gradient_variance': reference_variance,
                   'variance_ratio': current_variance / reference_variance if reference_variance > 1e-15 else None,
                   'interpretation': 'batch descriptive variance, not independent final-quality evidence'}
    return estimates.mean(axis=0), np.asarray(mean_advantages), diagnostics


def safe_layout_variant(text):
    # Avoid text-bearing macro arguments and control-symbol accents. This is a
    # limited layout probe, not a claim of general TeX semantic normalization.
    if re.search(r"\\(?:text|operatorname|mathrm|mbox)\b", text):
        return text
    return re.sub(r"\s*(?<!\\)([+=<>])\s*", r"\1", text)


class TerminalReward:
    def __init__(self, fit, reward, config):
        self.fit = fit
        self.reward = reward
        self.config = config
        self.structures = [structure_fast(r["latex"]) for r in reward]
        self.variants = [safe_layout_variant(r["latex"]) for r in reward]

    def score(self, model):
        # Use the actual exported encoder, not simulated replacement counts.
        fit_usage = Counter(id for r in self.fit for id in model.encode(r["latex"]))
        fit_total = sum(fit_usage.values())
        denominator = fit_total + .5 * len(model.vocab)
        groups = defaultdict(lambda: [0, 0])
        commands = command_ok = numbers = number_ok = crossing = tree_tokens = 0
        total = raw_bytes = rare = 0
        code_bits = robust = 0.
        for row, structure, variant in zip(self.reward, self.structures, self.variants):
            text = row["latex"]
            ids = model.encode(text)
            payload = [model.vocab[id] for id in ids]
            assert b"".join(payload) == text.encode()
            bounds = [0]
            for token in payload:
                bounds.append(bounds[-1] + len(token))
            length = len(text.encode())
            groups[row["split_group"]][0] += len(ids)
            groups[row["split_group"]][1] += length
            total += len(ids)
            raw_bytes += length
            for id in ids:
                code_bits -= math.log2((fit_usage[id] + .5) / denominator)
                rare += fit_usage[id] <= 5
            for a, b, token in structure.lexemes:
                intact = bisect_right(bounds, a) >= bisect_left(bounds, b)
                if token.startswith("\\") and token[1:].isalpha():
                    commands += 1
                    command_ok += intact
                if re.fullmatch(r"\d+(?:\.\d+)?", token):
                    numbers += 1
                    number_ok += intact
            if structure.kind != "byte_fallback":
                crossing += cross_count(bounds, structure.spans)
                tree_tokens += len(ids)
            original_content = sum(bool(token.strip()) for token in payload)
            variant_content = sum(bool(model.vocab[id].strip()) for id in model.encode(variant))
            robust += abs(original_content - variant_content) / max(1, original_content)
        stats = {
            "document_macro_tokens_per_byte": float(np.mean([t / max(1, b) for t, b in groups.values()])),
            "fit_predictive_unigram_bits_per_byte": code_bits / max(1, raw_bytes),
            "command_integrity": command_ok / commands if commands else 1.,
            "number_integrity": number_ok / numbers if numbers else 1.,
            "tree_crossing": crossing / max(1, tree_tokens),
            "layout_content_token_variation": robust / max(1, len(self.reward)),
            "fit_rare_token_rate": rare / max(1, total),
        }
        c = self.config
        stats["utility"] = (
            stats["document_macro_tokens_per_byte"]
            + c.code_weight * stats["fit_predictive_unigram_bits_per_byte"] / 8
            + c.command_weight * (1 - stats["command_integrity"])
            + c.number_weight * (1 - stats["number_integrity"])
            + c.tree_weight * stats["tree_crossing"]
            + c.robustness_weight * stats["layout_content_token_variation"]
            + c.rare_weight * stats["fit_rare_token_rate"]
        )
        return stats


class TrajectoryVocab(RLVocab):
    def __init__(self, config=None, trajectory=None):
        super().__init__(config)
        self.trajectory = trajectory or TrajectoryConfig()
        if self.trajectory.samples < 2 or self.trajectory.temperature <= 0:
            raise ValueError("invalid trajectory sampling configuration")
        if not 0 <= self.trajectory.exploration < 1:
            raise ValueError("invalid exploration mixture")
        if self.trajectory.baseline not in ('rloo', 'crossfit_critic'):
            raise ValueError('unknown trajectory baseline')
        self.theta = np.zeros(2 * len(self.theta))
        self.history = []

    def rollout(self, records, capacity, seed, theta, rng=None):
        model = self.make_rollout_model()
        session = MergeSession(model, records)
        score = np.zeros_like(theta)
        score_steps, critic_states = [], []
        entropy = 0.
        steps = max(0, capacity - len(model.vocab))
        for step in range(steps):
            pairs, features, rewards, components = session.candidates()
            if not pairs:
                break
            phase = 2 * step / max(1, steps - 1) - 1
            x = np.concatenate((features, features * phase), axis=1)
            if rng is None:
                action = int(np.argmax(x @ theta))
            else:
                p = probabilities_and_score(x, theta, temperature=self.trajectory.temperature, exploration=self.trajectory.exploration)
                action = int(rng.choice(len(p), p=p))
                _, local_score = probabilities_and_score(x, theta, action, self.trajectory.temperature, self.trajectory.exploration)
                score += local_score
                score_steps.append(local_score)
                critic_states.append(np.concatenate(([1., phase, session.totalheld / max(1, session.bytes_count)],
                                                      features.mean(axis=0), features.std(axis=0))))
                entropy -= float(np.sum(p * np.log(np.maximum(p, 1e-300))))
            # No policy update and no immediate reward gradient inside a rollout.
            session.commit(pairs[action], components[action], theta, np.max(rewards), "whole_trajectory")
        model.theta = theta.copy()
        model._trajectory_scores = np.asarray(score_steps)
        model._critic_states = np.asarray(critic_states)
        session.finish(capacity, seed, "rl_trajectory", 0.)
        return model, score, entropy / max(1, len(model.merges))

    def make_rollout_model(self):
        return RLVocab(self.config)

    def train(self, records, capacity=1024, seed=17, initial_theta=None, update=True, progress=None):
        started = time.perf_counter()
        if capacity < 256:
            raise ValueError("byte fallback requires capacity >=256")
        if initial_theta is not None:
            value = np.asarray(initial_theta, dtype=float)
            if len(value) == len(self.theta) // 2:
                value = np.concatenate((value, np.zeros_like(value)))
            if value.shape != self.theta.shape or not np.isfinite(value).all():
                raise ValueError("invalid initial policy")
            self.theta = value.copy()
        self.initial_theta = self.theta.copy()
        fit, reward = internal_split(records, self.config)
        objective = getattr(self,'terminal_reward_class',TerminalReward)(fit, reward, self.trajectory)
        initial, _, _ = self.rollout(records, capacity, seed, self.theta)
        self.initial_stats = objective.score(initial)
        anchor = max(1e-8, self.initial_stats["utility"])
        rng = np.random.default_rng(seed)
        first = np.zeros_like(self.theta)
        second = np.zeros_like(self.theta)
        self.history = []
        best = None
        best_utility = math.inf
        for epoch in range(self.trajectory.epochs):
            scores, returns, statistics, score_sequences, state_sequences = [], [], [], [], []
            frozen = self.theta.copy()
            for sample in range(self.trajectory.samples):
                model, score, entropy = self.rollout(records, capacity, seed, frozen, rng)
                stats = objective.score(model)
                value = (self.initial_stats["utility"] - stats["utility"]) / anchor
                scores.append(score)
                returns.append(value)
                statistics.append(dict(stats, relative_return=value, mean_action_entropy=entropy))
                score_sequences.append(model._trajectory_scores)
                state_sequences.append(model._critic_states)
                if stats["utility"] < best_utility:
                    best, best_utility = model, stats["utility"]
                if progress:
                    progress(epoch, sample, stats)
            gradient, advantages = rloo_gradient(scores, returns)
            critic = None
            if self.trajectory.baseline == 'crossfit_critic' and all(len(s) for s in state_sequences):
                gradient, advantages, critic = crossfit_critic_gradient(score_sequences, state_sequences, returns, self.trajectory.critic_ridge)
            norm = float(np.linalg.norm(gradient))
            if update:
                if norm > self.trajectory.gradient_clip:
                    gradient *= self.trajectory.gradient_clip / norm
                first = .9 * first + .1 * gradient
                second = .999 * second + .001 * gradient ** 2
                corrected = first / (1 - .9 ** (epoch + 1))
                variance = second / (1 - .999 ** (epoch + 1))
                self.theta += self.trajectory.learning_rate * corrected / (np.sqrt(variance) + 1e-8)
            self.history.append({"epoch": epoch, "samples": statistics,
                                 "gradient_norm_before_clip": norm, "advantages": advantages.tolist(),
                                 "policy": self.theta.tolist(), "updates_enabled": update,
                                 "reward_scope": "training_internal_document_holdout",
                                 "credit_assignment": "terminal_return_to_all_merge_actions_gamma_1"})
            self.history[-1]['baseline_estimator'] = self.trajectory.baseline
            if critic is not None:self.history[-1]['critic'] = critic
        # Deployment table is built deterministically from the final learned
        # policy. Best sampled trajectory stays a separately labeled search
        # control; it is never silently substituted for the trained policy.
        final, _, _ = self.rollout(records, capacity, seed, self.theta)
        self.final_stats = objective.score(final)
        self.vocab, self.merges, self.logs = final.vocab, final.merges, final.logs
        self.capacity, self.seed, self.method = capacity, seed, "rl_trajectory" if update else "trajectory_no_update"
        self.fit_ids, self.reward_ids = final.fit_ids, final.reward_ids
        self.train_seconds = time.perf_counter() - started
        self.best_sample = best
        if best is not None:best.train_seconds = self.train_seconds
        self.best_sample_stats = objective.score(best) if best is not None else None
        self.build_official()
        return self

    def save(self, path):
        super().save(path)
        saved = json.loads(path.read_text())
        saved.update(format="v5-offline-trajectory-rl", trajectory_config=asdict(self.trajectory),
                     initial_policy=self.initial_theta.tolist(), training_history=self.history,
                     initial_terminal_stats=self.initial_stats, final_terminal_stats=self.final_stats,
                     policy_update_scope="complete_merge_trajectory_terminal_return",
                     online_backend="unchanged_official_Rust_BPE")
        path.write_text(json.dumps(saved, indent=2))

    @classmethod
    def load(cls, path):
        saved = json.loads(path.read_text())
        obj = cls(VocabRLConfig(**saved["config"]), TrajectoryConfig(**saved["trajectory_config"]))
        obj.vocab = [bytes.fromhex(t) for t in saved["vocab_hex"]]
        obj.merges = [tuple(m) for m in saved["merges"]]
        obj.theta = np.asarray(saved["policy"])
        obj.logs, obj.history = saved["logs"], saved["training_history"]
        obj.initial_theta = np.asarray(saved["initial_policy"])
        obj.initial_stats, obj.final_stats = saved["initial_terminal_stats"], saved["final_terminal_stats"]
        for name in ("capacity", "seed", "method", "train_seconds", "fit_ids", "reward_ids"):
            setattr(obj, name, saved[name])
        obj.build_official()
        return obj
