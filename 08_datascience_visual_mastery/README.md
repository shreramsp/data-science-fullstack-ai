# 08 — Data Science Visual Foundations Curriculum

## Purpose

An interactive teaching curriculum for four foundational data science
concepts, aimed at beginner students, combining rigorous math with live,
adjustable simulations:

1. **Naive Bayes** — Bayes' theorem, the conditional-independence assumption,
   and a live 2-D Gaussian Naive Bayes decision-boundary simulation.
2. **Model evaluation** — confusion matrix, Type I / Type II errors, ROC-AUC,
   cost matrices, and the precision/recall tradeoff, via a live
   threshold-sweep simulation over synthetic classifier scores.
3. **Differential calculus → gradient descent** — the derivative as a tangent
   slope, and a live gradient-descent simulation on a non-convex 1-D curve.
4. **Chain rule → backpropagation** — a hand-derived, term-by-term chain-rule
   trace through a tiny one-hidden-unit network, plus a live training
   simulation (loss curve + learned fit).

Each concept includes a short quiz and a bank of interview-prep questions
with model answers.

## Setup

```bash
cd 08_datascience_visual_mastery
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

**Main interactive app (Streamlit — all four live simulations, full quiz
bank, full interview-prep bank):**

```bash
streamlit run app.py
```

**Static GitHub Pages companion (`docs/index.html`):** open the file
directly in a browser, or serve it locally:

```bash
python3 -m http.server 8000 --directory docs
```

To publish it on GitHub Pages: enable Pages for this repository pointed at
the `docs/` folder of this project (or copy `docs/index.html` to the
repository root / a dedicated Pages branch, per your GitHub Pages
configuration).

## Approach

- `src/naive_bayes.py`, `src/model_evaluation.py`,
  `src/calculus_gradient_descent.py`, `src/chain_rule_backprop.py` — small,
  pure functions implementing each concept's math directly (Bayes' theorem,
  confusion-matrix counting, an ROC curve swept from first principles, a
  hand-derived chain-rule backward pass), decoupled from the UI so the math
  can be checked independently of Streamlit.
- `src/content.py` — quiz questions and interview-prep Q&A as plain data.
- `app.py` — Streamlit app wiring sliders to the math functions so every
  chart updates live as a student changes a parameter (class means/spread,
  decision threshold, learning rate, network weights, etc.).
- `docs/index.html` — a single, dependency-free static page (vanilla
  HTML/CSS/JS, no build step, no external CDN) that can be hosted directly
  on GitHub Pages. It mirrors two of the four live simulations (model
  evaluation threshold sweep, gradient descent) in Canvas/JS, gives worked
  numeric examples for the other two (Naive Bayes, chain rule), and includes
  a short quiz and interview-prep list. The full four-simulation experience
  with the complete quiz and interview-prep banks lives in the Streamlit app.

## Honest results / validation performed

- Naive Bayes: verified the posterior correctly favors the class whose mean
  is closer to a test point (sanity check with symmetric classes).
- Model evaluation: verified AUC stays in `(0.5, 1.0]` for separable
  synthetic classes and that precision/recall respond to the threshold as
  expected.
- Calculus: verified the analytic derivative matches a central-difference
  numerical derivative to `<1e-8`.
- Chain rule: verified every hand-derived gradient (`dL/dw1`, `dL/db1`,
  `dL/dw2`, `dL/db2`) matches numerical differentiation of the loss to
  `<1e-4`, and confirmed training loss decreases over epochs on a toy
  `sin(x)` regression task.
- Confirmed `streamlit run app.py` starts cleanly (health check `ok`, no
  server-side exceptions) and confirmed `docs/index.html` serves and its
  inline JavaScript passes a syntax check.

**Limitations:** this is an educational simulation, not a production ML
library — Naive Bayes and the tiny backprop network use synthetic Gaussian
data rather than a real dataset, since the goal is building visual/mathematical
intuition for the mechanics themselves. The `docs/` static page intentionally
covers two live simulations plus worked examples rather than duplicating all
four Streamlit simulations, to avoid re-implementing the same interactivity
twice.
