# Chapter 8 — Expected Free Energy: epistemic value in a discrete cue-and-commit task

## Purpose

Chapter 7 demonstrated an Active-Inference-inspired perception–action loop in a
continuous thermostat environment. Chapter 8 moves from continuous regulation
to discrete policy selection.

The goal is to make one central idea testable:

> When the state of the world is uncertain, can an agent justify paying a small
> immediate cost to obtain a diagnostic observation before committing to an
> action?

The implementation will use a small, finite-horizon generative model. It is a
didactic demonstration of expected-free-energy-style policy evaluation, not a
complete biologically realistic Active Inference architecture.

## Research question

How do prior uncertainty, cue reliability, and inspection cost affect the
choice between immediate commitment and information-seeking under a discrete
expected-free-energy policy score?

## Environment

Each episode contains one hidden binary state:

```text
left_good
right_good
```

The agent begins with a prior belief:

```text
P(left_good) = p
P(right_good) = 1 - p
```

It can select one of three policies:

| Policy | Immediate consequence | Later consequence |
| --- | --- | --- |
| `commit_left` | No observation | Receives `success` when `left_good` is true, otherwise `failure` |
| `commit_right` | No observation | Receives `success` when `right_good` is true, otherwise `failure` |
| `inspect_then_commit` | Pays an inspection cost and observes a noisy binary cue | Updates its belief and commits to the more probable state |

The inspection cue has configurable reliability:

```text
P(cue = true_state) = reliability
```

For example, at `reliability = 0.85`, the cue identifies the hidden state
correctly in 85% of episodes.

## Generative-model ingredients

The initial version makes all modelling assumptions explicit:

- **Hidden state:** which side is currently good.
- **Prior belief:** probability that the left side is good before observing a
  cue.
- **Likelihood model:** probability of each cue under each hidden state.
- **Outcome preferences:** `success` is preferred to `failure`.
- **Inspection cost:** a transparent penalty for obtaining a cue.
- **Finite horizon:** inspect once, update beliefs once, then commit once.

After observing a cue, the agent uses Bayes' rule:

```text
P(state | cue) ∝ P(cue | state) P(state)
```

## Operational policy score

Expected free energy is often expressed as a trade-off between satisfying prior
preferences over outcomes and resolving uncertainty about hidden states.

For this finite, discrete model, the project uses the following higher-is-better
policy score:

```text
EFE-style policy score(policy)
    = expected preference value
    + epistemic_weight × expected information gain
    - inspection cost
```

The expected information gain of a policy is:

```text
E_cue [ KL( P(state | cue) || P(state) ) ]
```

where `KL` is the Kullback–Leibler divergence. It measures how much, on
average, observing the cue is expected to change the agent's belief about the
hidden state.

This score is an explicit pedagogical decomposition for the defined
two-step model. It is not presented as a general implementation of all
expected-free-energy formulations, variational inference, generalized
coordinates, or neural process theory.

## Comparison baseline

The benchmark will compare two planners over the same environment and policy
set:

| Planner | Policy score |
| --- | --- |
| `PreferenceOnlyPlanner` | Expected preference value minus inspection cost |
| `ExpectedFreeEnergyPlanner` | Expected preference value plus weighted expected information gain minus inspection cost |

The baseline is deliberately named **preference-only**, not reinforcement
learning. It does not learn values from experience; it evaluates the known
model without assigning intrinsic value to information.

## Hypotheses

1. **Uncertainty-sensitive inspection:** the EFE planner should inspect more
   frequently when the prior is uncertain, especially near
   `P(left_good) = 0.5`.

2. **Reliability-sensitive inspection:** inspection should become less likely
   as cue reliability approaches chance level.

3. **Certainty-sensitive commitment:** when the prior strongly favours one
   state, the EFE planner should commit directly because a cue has little
   expected information gain.

4. **Epistemic difference:** for suitable combinations of moderate inspection
   cost and reliable cues, the EFE planner should inspect in conditions where
   the preference-only planner commits immediately.

These are model predictions for a defined simulation, not claims about human
or animal behaviour.

## Experimental conditions

The initial sensitivity sweep will vary:

| Parameter | Example values |
| --- | --- |
| Prior probability that `left_good` | 0.50, 0.65, 0.80, 0.95 |
| Cue reliability | 0.50, 0.65, 0.80, 0.95 |
| Inspection cost | 0.00, 0.05, 0.10, 0.20 |
| Epistemic weight | 0.00, 0.25, 0.50, 1.00 |
| Random seeds | At least 64 per condition |

## Metrics

For each configuration and planner, record:

- selected policy;
- expected preference value;
- expected information gain;
- prior entropy;
- posterior entropy after inspection;
- realised success or failure;
- realised cumulative outcome after inspection cost;
- inspection rate across seeded episodes.

The key figures should show:

1. inspection rate as a function of prior uncertainty and cue reliability;
2. expected information gain across the same grid;
3. cumulative outcome comparison between planners;
4. representative belief updates for uncertain and near-certain priors.

## Implementation plan

### Phase 1 — probability primitives

Implement and test:

- binary entropy;
- Bayesian posterior update;
- Kullback–Leibler divergence;
- expected information gain under a binary cue model.

### Phase 2 — environment and policy evaluation

Implement:

- the hidden-state environment;
- policy definitions;
- expected preference values;
- the preference-only and EFE-style policy scores.

### Phase 3 — reproducible benchmark

Implement:

- configuration-driven sweeps;
- seeded episode simulation;
- tabular result export;
- sensitivity figures.

### Phase 4 — explanation and limits

Add:

- `notebooks/Chapter08.ipynb`;
- a chapter figure and interpretation;
- a limitations section in the README and chapter documentation;
- tests for every probability and policy-selection invariant.

## Acceptance criteria

Chapter 8 is ready for release only when:

- Bayesian posteriors are normalised and move toward the cue-consistent state;
- an uninformative cue with reliability `0.5` has expected information gain
  equal to zero;
- a deterministic prior has expected information gain