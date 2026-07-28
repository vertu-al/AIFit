# AIFit

> *Learn Active Inference by building it.*

AIFit is an open-source educational framework that implements the concepts of **Active Inference** in Python through transparent simulations, mathematical derivations, and reproducible code.

The project follows the book

> **Fundamentals of Active Inference: Principles, Algorithms, and Applications**

chapter by chapter, translating theoretical concepts into executable models.

The long-term vision is to provide a reusable Active Inference framework for computational neuroscience, adaptive systems, health, robotics, and machine learning.

---

# Philosophy

AIFit follows a simple learning cycle:

```text
Read
   ↓
Understand
   ↓
Implement
   ↓
Visualize
   ↓
Experiment
```

Rather than treating Active Inference as abstract mathematics, AIFit aims to make every major concept executable and reproducible.

---

# Current Progress

| Chapter | Status | Description |
|---------|--------|-------------|
| 7 | ✅ Complete | Bayesian Thermostat & Action–Perception Loop |
| 8 | ⬜ Planned | Expected Free Energy |
| 9 | ⬜ Planned | Policy Selection |
| 10 | ⬜ Planned | Learning |

---

# Repository Structure

```text
AIFit
│
├── src/
│   └── aifit/
│       ├── agent.py
│       ├── config.py
│       ├── environment.py
│       ├── plotting.py
│       └── simulation.py
│
├── tests/
│
├── notebooks/
│
├── figures/
│
├── docs/
│
└── README.md
```

---

# Current Example

## Bayesian Thermostat (Chapter 7)

The first implementation demonstrates the core Active Inference cycle.

```text
Environment
      ↓
Observation
      ↓
Prediction Error
      ↓
Belief Update
      ↓
Action
      ↓
Environment
```

The simulation illustrates how perception updates hidden-state beliefs while action changes the environment to fulfil preferred sensory states.

The generated figures reproduce the qualitative behaviour described in Chapter 7.

---

# Current Features

- Continuous hidden-state simulation
- Bayesian belief updates
- Active perception
- Active control
- Prediction errors
- Free-energy proxy
- Publication-quality figures
- Automated tests
- Reproducible simulation configuration

---

# Installation

Clone the repository

```bash
git clone https://github.com/vertu-al/AIFit.git
cd AIFit
```

Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install

```bash
pip install -e ".[dev]"
```

Run the simulation

```bash
python -m aifit.simulation
```

Run the tests

```bash
python -m pytest
```

---

# Roadmap

## Foundations

- ✅ Chapter 7
- ⬜ Chapter 8
- ⬜ Chapter 9
- ⬜ Chapter 10

## Framework

- Trajectory objects
- Proper variational free energy
- Generalized coordinates
- Expected free energy
- Policy selection
- Learning

## Example Library

- Bayesian Thermostat
- Mountain Car
- Pendulum
- Cart-Pole
- Fitness
- Homeostasis
- Cochlear Implant (planned)
- Active Vision

---

# Design Principles

Every implementation in AIFit should be

- mathematically transparent
- scientifically reproducible
- well documented
- extensively tested
- easy to read
- directly connected to the underlying theory

The goal is not merely to reproduce published algorithms, but to understand why they work.

---

# License

This project is currently under active development.

A license will be added before the first stable release.

---

# Citation

If you find AIFit useful for teaching or research, please cite the original Active Inference literature.

---

## Author

**Verena Uyka**

Biologist • Artificial Intelligence Student • Neuro- & Psycholinguistics (M.Sc.)

Research interests:

- Active Inference
- Computational Neuroscience
- Cochlear Implants
- Auditory Learning
- Predictive Processing