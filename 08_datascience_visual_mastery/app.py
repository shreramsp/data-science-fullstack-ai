"""Data Science Visual Foundations Curriculum.

An interactive Streamlit curriculum teaching four beginner-to-intermediate
data science concepts with live simulations: Naive Bayes, model evaluation
(confusion matrix / ROC / cost-sensitive thresholds), differential calculus
and gradient descent, and the chain rule behind backpropagation.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import calculus_gradient_descent as cgd
import chain_rule_backprop as crb
import model_evaluation as meval
import naive_bayes as nb
from content import INTERVIEW_QUESTIONS, QUIZZES

st.set_page_config(page_title="DS Visual Foundations", page_icon="📐", layout="wide")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COLOR_A, COLOR_B, COLOR_ACCENT = "#2a78d6", "#eb6834", "#1baf7a"


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


st.title("📐 Data Science Visual Foundations Curriculum")
st.caption(
    "An interactive curriculum for beginner "
    "data science students naive bayes, model evaluation, differential calculus / "
    "gradient descent, and the chain rule / backpropagation with deep intuition, "
    "rigorous math, and live simulation. This is an independent implementation."
)

tab_nb, tab_eval, tab_calc, tab_chain, tab_quiz, tab_interview = st.tabs([
    "1. Naive Bayes", "2. Model Evaluation", "3. Calculus & Gradient Descent",
    "4. Chain Rule & Backprop", "📝 Quizzes", "💼 Interview Prep",
])

# ---------------------------------------------------------------------------
# 1. Naive Bayes
# ---------------------------------------------------------------------------
with tab_nb:
    st.header("Naive Bayes")
    st.markdown(
        "Naive Bayes classifies a point by comparing, for each class, how "
        "plausible that point is under that class's distribution, weighted "
        "by how common the class is overall."
    )
    st.latex(r"P(\text{class} \mid x_1, x_2) = "
             r"\frac{P(x_1 \mid \text{class})\, P(x_2 \mid \text{class})\, "
             r"P(\text{class})}{P(x_1, x_2)}")
    st.markdown(
        "The **naive** part: it assumes $x_1$ and $x_2$ are *conditionally "
        "independent* given the class, so their joint likelihood is just the "
        "product of two 1-D Gaussians. That's rarely exactly true, but the "
        "classifier only needs to rank classes correctly, so it works well "
        "in practice."
    )

    st.subheader("Live simulation: two Gaussian classes in 2-D")
    col_a, col_b, col_test = st.columns(3)
    with col_a:
        st.markdown(f"**Class A** :large_blue_circle:")
        mean_a1 = st.slider("Class A mean (feature 1)", -4.0, 4.0, -1.5, key="a_m1")
        mean_a2 = st.slider("Class A mean (feature 2)", -4.0, 4.0, -1.0, key="a_m2")
        std_a = st.slider("Class A spread (std)", 0.3, 2.5, 1.0, key="a_std")
        prior_a = st.slider("Prior P(A)", 0.05, 0.95, 0.5, key="a_prior")
    with col_b:
        st.markdown(f"**Class B** :large_orange_circle:")
        mean_b1 = st.slider("Class B mean (feature 1)", -4.0, 4.0, 1.5, key="b_m1")
        mean_b2 = st.slider("Class B mean (feature 2)", -4.0, 4.0, 1.0, key="b_m2")
        std_b = st.slider("Class B spread (std)", 0.3, 2.5, 1.0, key="b_std")
        prior_b = 1.0 - prior_a
        st.metric("Prior P(B)", f"{prior_b:.2f}")
    with col_test:
        st.markdown("**Test point**")
        x1_test = st.slider("x1", -6.0, 6.0, 0.0, key="test_x1")
        x2_test = st.slider("x2", -6.0, 6.0, 0.0, key="test_x2")

    mean_a, mean_b = (mean_a1, mean_a2), (mean_b1, mean_b2)
    std_a_t, std_b_t = (std_a, std_a), (std_b, std_b)

    p_a, p_b = nb.posterior(x1_test, x2_test, mean_a, std_a_t, prior_a,
                             mean_b, std_b_t, prior_b)
    pred = "A" if p_a >= p_b else "B"

    m1, m2, m3 = st.columns(3)
    m1.metric("P(A | x)", f"{p_a:.3f}")
    m2.metric("P(B | x)", f"{p_b:.3f}")
    m3.metric("Predicted class", pred)

    xx, yy, dens_a, dens_b, post_a = nb.decision_grid(
        mean_a, std_a_t, prior_a, mean_b, std_b_t, prior_b)
    sample_a, sample_b = nb.sample_dataset(mean_a, std_a_t, 120, mean_b, std_b_t, 120)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(xx, yy, dens_a, levels=5, colors=COLOR_A, alpha=0.6)
    ax.contour(xx, yy, dens_b, levels=5, colors=COLOR_B, alpha=0.6)
    ax.contourf(xx, yy, post_a, levels=[0, 0.5, 1], colors=[COLOR_B, COLOR_A], alpha=0.08)
    ax.scatter(sample_a[:, 0], sample_a[:, 1], s=10, color=COLOR_A, alpha=0.5, label="Class A samples")
    ax.scatter(sample_b[:, 0], sample_b[:, 1], s=10, color=COLOR_B, alpha=0.5, label="Class B samples")
    ax.scatter([x1_test], [x2_test], s=180, marker="*", color=COLOR_ACCENT,
               edgecolor=INK, linewidth=1, zorder=5, label="Test point")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title("Class-conditional densities, decision region, and test point")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    _style_axes(ax)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Model Evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.header("Evaluating a classifier")
    st.markdown(
        "Every binary classifier makes two kinds of mistakes. Moving the "
        "decision threshold trades one off against the other -- it never "
        "eliminates both at once."
    )
    col_type1, col_type2 = st.columns(2)
    col_type1.info("**Type I error (False Positive):** predicting positive "
                    "when the truth is negative.")
    col_type2.warning("**Type II error (False Negative):** predicting "
                       "negative when the truth is positive.")

    st.subheader("Live simulation: threshold sweep on classifier scores")
    c1, c2, c3, c4 = st.columns(4)
    separation = c1.slider("Class separability", 0.3, 3.0, 1.5, 0.1)
    threshold = c2.slider("Decision threshold", -3.0, 4.0, 0.5, 0.05)
    cost_fp = c3.slider("Cost of a False Positive", 0.0, 10.0, 1.0, 0.5)
    cost_fn = c4.slider("Cost of a False Negative", 0.0, 10.0, 1.0, 0.5)

    y_true, y_score = meval.generate_scores(separation=separation)
    tp, fp, fn, tn = meval.confusion_counts(y_true, y_score, threshold)
    m = meval.metrics_from_counts(tp, fp, fn, tn)
    cost = meval.expected_cost(tp, fp, fn, tn, cost_fp, cost_fn)
    fprs, tprs, auc = meval.roc_curve(y_true, y_score)

    st.markdown("**Confusion matrix**")
    cm_cols = st.columns(2)
    with cm_cols[0]:
        st.table({
            "Predicted Positive": [f"TP = {tp}", f"FP = {fp}  (Type I)"],
            "Predicted Negative": [f"FN = {fn}  (Type II)", f"TN = {tn}"],
        })
    with cm_cols[1]:
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("Precision", f"{m['precision']:.2f}")
        mm2.metric("Recall", f"{m['recall']:.2f}")
        mm3.metric("F1", f"{m['f1']:.2f}")
        mm4, mm5, mm6 = st.columns(3)
        mm4.metric("Accuracy", f"{m['accuracy']:.2f}")
        mm5.metric("AUC", f"{auc:.2f}")
        mm6.metric("Expected cost", f"{cost:.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(y_score.min(), y_score.max(), 40)
    ax1.hist(y_score[y_true == 0], bins=bins, alpha=0.6, color=COLOR_B, label="Negative class")
    ax1.hist(y_score[y_true == 1], bins=bins, alpha=0.6, color=COLOR_A, label="Positive class")
    ax1.axvline(threshold, color=INK, linestyle="--", label="Threshold")
    ax1.set_title("Score distributions & threshold"); ax1.set_xlabel("score"); ax1.legend(fontsize=8)
    _style_axes(ax1)

    ax2.plot(fprs, tprs, color=COLOR_ACCENT, linewidth=2)
    ax2.plot([0, 1], [0, 1], color=MUTED, linestyle=":")
    op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    ax2.scatter([op_fpr], [m["recall"]], color=INK, zorder=5, label="Current threshold")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate (Recall)")
    ax2.set_title(f"ROC curve (AUC = {auc:.2f})"); ax2.legend(fontsize=8)
    _style_axes(ax2)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Precision/recall tradeoff: raising the threshold usually raises "
        "precision (fewer, more confident positive calls) but lowers recall "
        "(more real positives missed) -- watch both metrics move as you drag "
        "the threshold slider."
    )

# ---------------------------------------------------------------------------
# 3. Calculus & Gradient Descent
# ---------------------------------------------------------------------------
with tab_calc:
    st.header("Derivatives and gradient descent")
    st.markdown("A derivative is the slope of the tangent line to a curve at a point:")
    st.latex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}")
    st.markdown(
        "Gradient descent uses that slope to decide which way is 'downhill' "
        "and takes a small step in that direction:"
    )
    st.latex(r"x_{t+1} = x_t - \eta \, f'(x_t)")

    st.subheader("Live simulation")
    c1, c2, c3, c4 = st.columns(4)
    x0 = c1.slider("Tangent point x0", -3.5, 3.5, 1.0, 0.1)
    start_x = c2.slider("Gradient descent start x", -3.5, 3.5, -3.0, 0.1)
    lr = c3.slider("Learning rate", 0.01, 0.9, 0.15, 0.01)
    steps = c4.slider("Steps", 1, 40, 15)

    st.metric("f'(x0) analytic", f"{cgd.f_prime(x0):.4f}")
    st.metric("f'(x0) numerical (central difference)", f"{cgd.numerical_derivative(x0):.4f}")

    xs = np.linspace(-3.5, 3.5, 300)
    path = cgd.gradient_descent(start_x, lr, steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(xs, cgd.f(xs), color=INK)
    tangent_xs = np.linspace(x0 - 1, x0 + 1, 10)
    ax1.plot(tangent_xs, cgd.tangent_line(x0, tangent_xs), color=COLOR_ACCENT, linewidth=2)
    ax1.scatter([x0], [cgd.f(x0)], color=COLOR_ACCENT, zorder=5)
    ax1.set_title("f(x) with tangent line at x0"); ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")
    _style_axes(ax1)

    path_arr = np.array(path)
    ax2.plot(xs, cgd.f(xs), color=INK, alpha=0.4)
    ax2.plot(path_arr, cgd.f(path_arr), "o-", color=COLOR_A, markersize=4)
    ax2.scatter([path_arr[0]], [cgd.f(path_arr[0])], color=COLOR_ACCENT, s=90, zorder=5, label="Start")
    ax2.scatter([path_arr[-1]], [cgd.f(path_arr[-1])], color=COLOR_B, s=90, zorder=5, label="End")
    ax2.set_title("Gradient descent trajectory"); ax2.set_xlabel("x"); ax2.legend(fontsize=8)
    _style_axes(ax2)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"After {steps} steps with learning rate {lr}, x moved from "
        f"{path[0]:.3f} to {path[-1]:.3f} (f(x): {cgd.f(path[0]):.3f} -> "
        f"{cgd.f(path[-1]):.3f}). Try a large learning rate to see it "
        "overshoot or oscillate."
    )

# ---------------------------------------------------------------------------
# 4. Chain Rule & Backprop
# ---------------------------------------------------------------------------
with tab_chain:
    st.header("Chain rule and backpropagation")
    st.latex(r"\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)")
    st.markdown(
        "A tiny network is just a composition of functions: "
        "$z_1 = w_1 x + b_1$, $a_1 = \\tanh(z_1)$, $\\hat{y} = w_2 a_1 + b_2$, "
        "$L = \\tfrac{1}{2}(\\hat{y}-y)^2$. Backprop applies the chain rule "
        "once per layer, back to front, reusing each intermediate gradient."
    )

    st.subheader("Trace the chain rule for one example")
    c1, c2, c3, c4, c5 = st.columns(5)
    x_ex = c1.slider("x", -2.0, 2.0, 1.0, 0.1)
    y_ex = c2.slider("target y", -2.0, 2.0, 0.5, 0.1)
    w1_ex = c3.slider("w1", -2.0, 2.0, 0.8, 0.1)
    w2_ex = c4.slider("w2", -2.0, 2.0, -0.6, 0.1)
    b1_ex = c5.slider("b1, b2 (shared)", -2.0, 2.0, 0.0, 0.1)

    cache = crb.forward(x_ex, y_ex, w1_ex, b1_ex, w2_ex, b1_ex)
    grads = crb.backward(cache)

    st.markdown("**Forward pass**")
    st.write({k: round(cache[k], 4) for k in ["z1", "a1", "yhat", "loss"]})
    st.markdown("**Backward pass (chain rule, term by term)**")
    st.latex(r"\frac{dL}{dw_1} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{da_1} "
             r"\cdot \frac{da_1}{dz_1} \cdot \frac{dz_1}{dw_1}")
    st.write({
        "dL/dyhat": round(grads["dL_dyhat"], 4),
        "dL/da1 (= dL/dyhat * w2)": round(grads["dL_da1"], 4),
        "da1/dz1 (= 1 - tanh(z1)^2)": round(grads["da1_dz1"], 4),
        "dL/dz1": round(grads["dL_dz1"], 4),
        "dL/dw1 (final product)": round(grads["dL_dw1"], 4),
    })

    st.subheader("Live simulation: training the tiny network")
    tc1, tc2 = st.columns(2)
    train_lr = tc1.slider("Training learning rate", 0.01, 1.0, 0.3, 0.01)
    train_epochs = tc2.slider("Epochs", 10, 500, 150, 10)

    x_data, y_data = crb.toy_dataset()
    result = crb.train(x_data, y_data, train_lr, train_epochs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(result["loss_history"], color=COLOR_A)
    ax1.set_title("Training loss over epochs"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    _style_axes(ax1)

    xs_plot = np.linspace(-3, 3, 200)
    preds = [crb.forward(xv, 0, result["w1"], result["b1"], result["w2"], result["b2"])["yhat"]
             for xv in xs_plot]
    ax2.scatter(x_data, y_data, s=15, color=MUTED, label="Training data (sin(x)+noise)")
    ax2.plot(xs_plot, preds, color=COLOR_ACCENT, linewidth=2, label="Learned fit")
    ax2.set_title("Fit after training"); ax2.legend(fontsize=8)
    _style_axes(ax2)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"Final loss: {result['loss_history'][-1]:.4f}. A single tanh unit "
        "can only approximate part of sin(x)'s curvature -- that's an honest "
        "limitation of a 1-hidden-unit network, not a bug."
    )

# ---------------------------------------------------------------------------
# 5. Quizzes
# ---------------------------------------------------------------------------
with tab_quiz:
    st.header("Concept quizzes")
    for concept, questions in QUIZZES.items():
        st.subheader(concept)
        for i, q in enumerate(questions):
            key = f"quiz_{concept}_{i}"
            choice = st.radio(q["question"], q["options"], index=None, key=key)
            if choice is not None:
                selected_idx = q["options"].index(choice)
                if selected_idx == q["answer"]:
                    st.success(f"Correct. {q['explanation']}")
                else:
                    st.error(f"Not quite -- correct answer: "
                              f"**{q['options'][q['answer']]}**. {q['explanation']}")
        st.divider()

# ---------------------------------------------------------------------------
# 6. Interview Prep
# ---------------------------------------------------------------------------
with tab_interview:
    st.header("Interview prep questions")
    for concept, questions in INTERVIEW_QUESTIONS.items():
        st.subheader(concept)
        for q in questions:
            with st.expander(q["q"]):
                st.write(q["a"])
        st.divider()
