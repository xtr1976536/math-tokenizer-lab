import itertools
import json
import math
import random

import numpy as np
import pytest
from tokenizers import Tokenizer

from math_tokenizer.ast import compatible
from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.checkpoints import resolve_checkpoint, verify_saved_hash
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.corpus import ROOT
from math_tokenizer.current_models import load_checkpoint, load_current
from math_tokenizer.dictionary_v4 import augment_dictionary
from math_tokenizer.rl_vocab_v5 import Indexed, RLVocab, VocabRLConfig, internal_split
from math_tokenizer.trajectory_v5 import (
    TerminalReward, TrajectoryConfig, TrajectoryVocab,
    probabilities_and_score, rloo_gradient, safe_layout_variant, crossfit_critic_gradient,
)


def test_current_registry_and_retirement_scope():
    active = json.loads((ROOT / 'models/current/registry.json').read_text())
    assert active['families'] == ['ast_v4', 'v5']
    retired = json.loads((ROOT / 'models/retirement_manifest.json').read_text())
    assert retired['removed_files'] == len(retired['files'])
    for relative, entry in retired['files'].items():
        assert relative.startswith('results/') and '/models/' in relative
        assert not (ROOT / relative).exists()
        state = verify_saved_hash(ROOT / relative, entry['sha256'])
        assert state == ('relocated' if entry['relocated_to'] else 'retired')
    for path in active['ast_v4'].values():
        assert json.loads((ROOT / path).read_text())['inference'] == 'shortest'
    with pytest.raises(AssertionError):
        verify_saved_hash(ROOT / 'results/unknown/models/not-recorded.json', '0' * 64)


def test_relocated_checkpoint_corruption_is_not_exempted():
    original = ROOT / 'results/v4_1024/models/ast-atom256.json'
    assert resolve_checkpoint(original) == ROOT / 'models/current/ast_v4/base-1024.json'
    with pytest.raises(AssertionError):
        verify_saved_hash(original, '0' * 64)


def test_current_loader_accepts_string_paths_and_both_families():
    registry = json.loads((ROOT / 'models/current/registry.json').read_text())
    for family in ('ast_v4', 'v5'):
        a = load_current(family)
        relative = registry['ast_v4']['1024'] if family == 'ast_v4' else registry['v5']['broad']
        b = load_checkpoint(str(ROOT / relative))
        text = r'\frac{x+y}{z}=\alpha'
        assert a.encode(text) == b.encode(text)
        assert a.decode(a.encode(text)) == text


def test_indexed_gains_and_heap_match_independent_replacements():
    texts = ['aaaaaaa+bbbb+ababab', r'\alpha+\alpha', '中文🙂+x+x', 'aaaaabaaaa']
    cuts = [set(range(1, len(t.encode()))) for t in texts]
    model = Indexed(texts, cuts)
    for step in range(32):
        reference = {}
        for seq in model.seqs:
            for pair, count in seq.counts().items():
                reference[pair] = reference.get(pair, 0) + count
        ordered = sorted((p for p, c in reference.items() if c >= 1), key=lambda p: (-reference[p], p))[:12]
        assert model.top(12, 1) == ordered
        for pair in ordered:
            assert model.gains[pair] == sum(s.gain(pair) for s in model.seqs)
            expected_boundary = 0
            for seq, positions in zip(model.seqs, cuts):
                i = 0
                while i < len(seq.ids) - 1:
                    if (seq.ids[i], seq.ids[i + 1]) == pair:
                        expected_boundary += seq.ends[i] in positions
                        i += 2
                    else:
                        i += 1
            assert model.boundaries[pair] == expected_boundary
        if not ordered:
            break
        model.merge(ordered[step % len(ordered)], 256 + step)


@pytest.mark.parametrize('hard', [False, True])
def test_ast_shortest_matches_exhaustive_legal_paths(hard):
    vocab = [bytes([b]) for b in range(256)] + [b'ab', b'bc', b'abc', b'cd', b'abcd']
    fast = ConstrainedTokenizer(vocab, hard, mode='shortest')
    reference = ConstrainedTokenizer(vocab, hard, mode='shortest', shortest_backend='reference')
    rng = random.Random(17)
    for _ in range(40):
        text = 'abcdab'
        raw = text.encode()
        spans = sorted({(a, b) for a in range(6) for b in range(a + 1, 7) if rng.random() < .12})
        span_array = np.array(spans, dtype=np.int32).reshape(-1, 2)
        allowed = np.ones(7, dtype=np.uint8)
        safe = np.zeros(7, dtype=np.uint8)
        interior = np.zeros(7, dtype=np.uint8)
        def paths(start):
            if start == len(raw):
                return [[]]
            result = []
            for id, token in enumerate(vocab):
                end = start + len(token)
                if raw.startswith(token, start) and (not hard or compatible(start, end, spans)):
                    result += [[id] + tail for tail in paths(end)]
            return result
        choices = paths(0)
        ids = fast.encode_prepared4(raw, span_array, safe, interior, allowed)
        assert ids == reference.encode_prepared4(raw, span_array, safe, interior, allowed)
        assert len(ids) == min(map(len, choices))
        assert fast.decode(ids) == text


def test_mixture_policy_score_and_full_trajectory_rloo_gradient():
    theta = np.array([.2, -.3])
    first = np.array([[1., -.2], [-.4, .7]])
    second = [np.array([[.2, .8], [.6, -.5]]), np.array([[1., .2], [-.3, -.6]])]
    terminal = np.array([[.8, -.1], [.2, .5]])
    def enumerate_paths(value):
        out = []
        p = probabilities_and_score(first, value, temperature=1.7, exploration=.1)
        for a, b in itertools.product(range(2), repeat=2):
            q = probabilities_and_score(second[a], value, temperature=1.7, exploration=.1)
            _, sa = probabilities_and_score(first, value, a, 1.7, .1)
            _, sb = probabilities_and_score(second[a], value, b, 1.7, .1)
            out.append((p[a] * q[b], terminal[a, b], sa + sb))
        return out
    paths = enumerate_paths(theta)
    exact = sum(p * reward * score for p, reward, score in paths)
    finite = []
    for k in range(2):
        delta = np.zeros(2); delta[k] = 1e-5
        plus = sum(p * r for p, r, s in enumerate_paths(theta + delta))
        minus = sum(p * r for p, r, s in enumerate_paths(theta - delta))
        finite.append((plus - minus) / 2e-5)
    assert np.allclose(exact, finite, atol=1e-9)
    estimate = np.zeros(2)
    for (p, r, s), (q, u, t) in itertools.product(paths, repeat=2):
        gradient, _ = rloo_gradient([s, t], [r, u])
        estimate += p * q * gradient
    assert np.allclose(estimate, exact, atol=1e-12)
    # Learned state baselines train on the OTHER trajectory. Enumerating the
    # complete paired experiment verifies that this cross-fitting stays
    # unbiased, including dependence of later states on earlier actions.
    actions = list(itertools.product(range(2), repeat=2))
    score_steps, states = [], []
    for a, b in actions:
        _, sa = probabilities_and_score(first, theta, a, 1.7, .1)
        _, sb = probabilities_and_score(second[a], theta, b, 1.7, .1)
        score_steps.append(np.array([sa, sb]))
        states.append(np.array([[1., 0., .5], [1., 1., float(a)]]))
    crossfit = np.zeros(2)
    for i, j in itertools.product(range(4), repeat=2):
        gradient, _, diagnostic = crossfit_critic_gradient([score_steps[i], score_steps[j]], [states[i], states[j]], [paths[i][1], paths[j][1]])
        assert all(m['excluded_rollout'] not in m['trained_on_rollouts'] for m in diagnostic['cross_fitted_value_heads'])
        crossfit += paths[i][0] * paths[j][0] * gradient
    assert np.allclose(crossfit, exact, atol=1e-10)


def test_subtree_vocabulary_budget_constraints_and_unseen_fallback():
    texts = [r'\frac{x+y}{z}=1', r'\alpha+(x+y)=z', '(x+y)^2=x^2+2xy+y^2'] * 4
    trainer = GrammarBPE(True).train(texts, 300, atom_budget=16, subtree_budget=12)
    log = trainer.training_log[0]
    assert 0 < log['reserved_subtrees'] <= 12
    assert len(trainer.vocab) <= 300
    assert len(set(trainer.vocab)) == len(trainer.vocab)
    assert trainer.vocab[:256] == [bytes([b]) for b in range(256)]
    model = ConstrainedTokenizer(trainer.vocab, True, mode='shortest')
    for text in texts + [r'\frac{a+b}{c}=2', '中文🙂+x', r'\newmacro{x}', r'\frac{']:
        ids = model.encode(text)
        assert model.decode(ids) == text
        raw, spans, safe, interior, allowed, exceptions, structure = model.prepare(text)
        start = 0
        for id in ids:
            end = start + len(model.vocab[id])
            assert allowed[start] and allowed[end]
            if structure.kind != 'byte_fallback':
                assert compatible(start, end, spans)
            start = end


def test_dictionary_augmentation_preserves_grammar_and_held_paper_scope():
    records = toy_records()
    cfg = VocabRLConfig(fit_limit=100, reward_limit=30)
    fit, reward = internal_split(records, cfg)
    trainer = GrammarBPE(True).train([r['latex'] for r in fit], 300, atom_budget=16)
    base = ConstrainedTokenizer(trainer.vocab, True, mode='shortest')
    base.training_log = trainer.training_log
    base.train_seconds = trainer.train_seconds
    model = augment_dictionary(base, fit, reward, 300, extra_subtrees=32)
    assert len(model.vocab) <= 300
    assert model.vocab[:256] == [bytes([b]) for b in range(256)]
    original_atoms = {bytes.fromhex(t) for t in trainer.training_log[0]['selected_atom_hex']}
    assert original_atoms <= set(model.vocab)
    log = model.training_log[-1]
    assert set(log['fit_ids']).isdisjoint(log['reward_ids'])
    for text in [r'\alpha+\beta=1', r'\frac{z}{w}', '中文🙂+x', r'\unknown{x}']:
        assert model.decode(model.encode(text)) == text


def toy_records():
    texts = [r'\alpha+\beta=\frac{x}{y}', 'x+x+x+x=4x', '(a+b)^2=a^2+2ab+b^2', 'α+β=γ'] * 12
    return [dict(id=str(i), latex=t, split_group=f'paper{i//4}', domain='analysis') for i, t in enumerate(texts)]


def test_terminal_trajectory_learning_and_unmodified_official_deployment(tmp_path):
    records = toy_records()
    cfg = VocabRLConfig(fit_limit=100, reward_limit=30, updates=2)
    trajectory = TrajectoryConfig(epochs=2, samples=3, temperature=2.)
    model = TrajectoryVocab(cfg, trajectory).train(records, 288, 17)
    assert any(abs(v) > 1e-8 for v in model.theta)
    assert len(model.history) == 2
    assert all(len(r['samples']) == 3 for r in model.history)
    assert all(r['credit_assignment'] == 'terminal_return_to_all_merge_actions_gamma_1' for r in model.history)
    assert set(model.fit_ids).isdisjoint(model.reward_ids)
    path = tmp_path / 'model.json'
    model.save(path)
    loaded = TrajectoryVocab.load(path)
    official = Tokenizer.from_file(str(path.with_suffix('.tokenizer.json')))
    for text in ['', '中文🙂+1', r'\unknown{\alpha}', '\n x + y \t', '9' * 300]:
        ids = model.encode(text)
        assert loaded.encode(text) == official.encode(text, add_special_tokens=False).ids == ids
        assert model.decode(ids) == text
    fit, reward = internal_split(records, cfg)
    stats = TerminalReward(fit, reward, trajectory).score(model)
    assert stats['fit_predictive_unigram_bits_per_byte'] > 0
    assert stats['utility'] > stats['document_macro_tokens_per_byte']


def test_no_update_control_freezes_policy_and_layout_probe_is_limited():
    model = TrajectoryVocab(VocabRLConfig(fit_limit=100, reward_limit=30), TrajectoryConfig(epochs=1, samples=2))
    model.train(toy_records(), 280, 29, update=False)
    assert np.array_equal(model.theta, model.initial_theta)
    assert not model.history[0]['updates_enabled']
    assert safe_layout_variant(r'\alpha  =  x') == r'\alpha=x'
    assert safe_layout_variant(r'\alpha x') == r'\alpha x'
    assert safe_layout_variant(r'\text{a = b}') == r'\text{a = b}'
    assert safe_layout_variant(r'\= a') == r'\= a'


def test_critic_control_keeps_rollout_search_identical_without_actor_updates():
    cfg = VocabRLConfig(fit_limit=100, reward_limit=30)
    base = TrajectoryVocab(cfg, TrajectoryConfig(epochs=1, samples=3)).train(toy_records(), 280, 17, update=False)
    critic = TrajectoryVocab(cfg, TrajectoryConfig(epochs=1, samples=3, baseline='crossfit_critic')).train(toy_records(), 280, 17, update=False)
    assert base.merges == critic.merges and base.vocab == critic.vocab
    assert base.best_sample.merges == critic.best_sample.merges
    assert np.array_equal(base.theta, critic.theta)
    assert critic.history[0]['critic']['cross_fitted_value_heads']


def test_v5_retraining_starts_a_fresh_budget():
    model = RLVocab(VocabRLConfig(fit_limit=100, reward_limit=30, updates=2)).train(toy_records(), 288, 17)
    original = (model.vocab.copy(), model.merges.copy(), model.theta.copy())
    model.train(toy_records(), 288, 17)
    assert model.vocab == original[0] and model.merges == original[1]
    assert np.array_equal(model.theta, original[2])
