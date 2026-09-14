"""Static content: per-concept quizzes and interview-prep questions.

Kept as plain data (list of dicts) so app.py only has to render it — no
content lives inside the Streamlit view logic.
"""
from __future__ import annotations

QUIZZES: dict[str, list[dict]] = {
    "Naive Bayes": [
        {
            "question": "What does the 'naive' in Naive Bayes assume?",
            "options": [
                "Features are conditionally independent given the class",
                "The classes are linearly separable",
                "The data has no noise",
                "All features follow a uniform distribution",
            ],
            "answer": 0,
            "explanation": "Naive Bayes assumes p(x1, x2, ... | class) factorizes "
                            "into the product of per-feature likelihoods, which is "
                            "rarely exactly true but works well in practice.",
        },
        {
            "question": "Bayes' theorem computes P(class | x) using which quantities?",
            "options": [
                "Only the prior P(class)",
                "Likelihood P(x | class), prior P(class), and evidence P(x)",
                "Only the likelihood P(x | class)",
                "The mean and variance of x alone",
            ],
            "answer": 1,
            "explanation": "P(class|x) = P(x|class) P(class) / P(x); the evidence "
                            "P(x) is the normalizing constant across all classes.",
        },
        {
            "question": "If a class's prior probability increases while likelihoods "
                        "stay fixed, the posterior for that class will:",
            "options": ["Decrease", "Stay exactly the same", "Increase",
                        "Become undefined"],
            "answer": 2,
            "explanation": "Posterior is proportional to prior x likelihood, so a "
                            "larger prior directly increases the posterior.",
        },
    ],
    "Model Evaluation": [
        {
            "question": "A Type I error corresponds to which confusion-matrix cell?",
            "options": ["True Positive", "False Positive", "False Negative",
                        "True Negative"],
            "answer": 1,
            "explanation": "Type I error = false positive: predicting positive "
                            "when the true label is negative.",
        },
        {
            "question": "Raising the decision threshold generally makes precision:",
            "options": ["Go up (fewer, more confident positive predictions)",
                        "Go down", "Stay identical", "Undefined"],
            "answer": 0,
            "explanation": "A higher threshold means only very confident cases are "
                            "predicted positive, which typically raises precision "
                            "while lowering recall -- the classic tradeoff.",
        },
        {
            "question": "What does the ROC curve plot?",
            "options": ["Precision vs Recall", "True Positive Rate vs False "
                        "Positive Rate", "Accuracy vs Threshold",
                        "Cost vs Threshold"],
            "answer": 1,
            "explanation": "ROC = TPR (recall) on the y-axis against FPR on the "
                            "x-axis, swept across all thresholds; AUC is the area "
                            "under this curve.",
        },
        {
            "question": "In a cost matrix where false negatives are far more "
                        "expensive than false positives (e.g. missing cancer "
                        "diagnoses), the optimal threshold should be:",
            "options": ["Higher than the accuracy-optimal threshold",
                        "Lower than the accuracy-optimal threshold",
                        "Exactly 0.5 always", "Irrelevant to cost"],
            "answer": 1,
            "explanation": "To catch more true positives (reduce costly false "
                            "negatives) you lower the threshold, accepting more "
                            "false positives in exchange for fewer false negatives.",
        },
    ],
    "Calculus & Gradient Descent": [
        {
            "question": "The derivative f'(x) at a point represents:",
            "options": ["The value of f at x", "The slope of the tangent line "
                        "to f at x", "The area under f up to x",
                        "The average of f over all x"],
            "answer": 1,
            "explanation": "f'(x) = lim h->0 [f(x+h) - f(x)] / h, which is exactly "
                            "the instantaneous slope, i.e. the tangent line's slope.",
        },
        {
            "question": "Gradient descent updates a parameter using which rule?",
            "options": ["x_new = x_old + lr * f(x)", "x_new = x_old - lr * f'(x)",
                        "x_new = x_old * f'(x)", "x_new = f'(x)"],
            "answer": 1,
            "explanation": "We step in the direction opposite the gradient "
                            "(downhill) scaled by the learning rate.",
        },
        {
            "question": "If the learning rate is too large, gradient descent can:",
            "options": ["Converge faster with no downside", "Overshoot the "
                        "minimum and diverge or oscillate", "Always find the "
                        "global minimum", "Ignore the derivative entirely"],
            "answer": 1,
            "explanation": "Too-large steps can jump past the minimum repeatedly, "
                            "causing oscillation or divergence instead of "
                            "convergence.",
        },
    ],
    "Chain Rule & Backprop": [
        {
            "question": "The chain rule states that d/dx f(g(x)) equals:",
            "options": ["f'(x) * g'(x)", "f'(g(x)) * g'(x)", "f(g'(x))",
                        "f'(g(x)) + g'(x)"],
            "answer": 1,
            "explanation": "d/dx f(g(x)) = f'(g(x)) * g'(x) -- the derivative of "
                            "the outer function (evaluated at the inner function's "
                            "output) times the derivative of the inner function.",
        },
        {
            "question": "In backpropagation, gradients are computed:",
            "options": ["Layer by layer from the input to the output",
                        "Layer by layer from the output back to the input, "
                        "reusing the chain rule",
                        "All at once with no ordering",
                        "Only for the final layer"],
            "answer": 1,
            "explanation": "Backprop applies the chain rule starting at the loss "
                            "and moving backward, reusing each layer's local "
                            "gradient (dL/da) to compute the next one.",
        },
        {
            "question": "Why is backprop more efficient than computing each "
                        "weight's gradient independently from scratch?",
            "options": ["It isn't, it's just simpler to code",
                        "It reuses intermediate derivative terms across layers "
                        "instead of recomputing them",
                        "It skips the chain rule entirely",
                        "It only works for linear models"],
            "answer": 1,
            "explanation": "Backprop caches upstream gradients (e.g. dL/da1) and "
                            "reuses them for every downstream weight, avoiding "
                            "redundant computation -- this is what makes training "
                            "deep networks tractable.",
        },
    ],
}

INTERVIEW_QUESTIONS: dict[str, list[dict]] = {
    "Naive Bayes": [
        {"q": "Why does Naive Bayes still work well even though its "
              "independence assumption is almost always violated in practice?",
         "a": "It only needs to rank classes correctly, not estimate the "
              "likelihood exactly -- correlated features get 'double counted' "
              "similarly across classes, so the ranking (argmax) is often "
              "preserved even when the absolute probabilities are biased."},
        {"q": "How do you handle a feature value that never appeared with a "
              "given class in training (zero-frequency problem)?",
         "a": "Apply Laplace (additive) smoothing: add a small constant (e.g. "
              "1) to every count so no likelihood is exactly zero."},
        {"q": "What's the difference between Gaussian, Multinomial, and "
              "Bernoulli Naive Bayes?",
         "a": "They differ in the assumed likelihood distribution: Gaussian "
              "for continuous features, Multinomial for count data (e.g. word "
              "counts), Bernoulli for binary presence/absence features."},
    ],
    "Model Evaluation": [
        {"q": "Why is accuracy a misleading metric on an imbalanced dataset?",
         "a": "A model that always predicts the majority class can get very "
              "high accuracy while having zero recall on the minority class, "
              "which is usually the class you actually care about."},
        {"q": "When would you optimize for recall over precision, and vice versa?",
         "a": "Optimize recall when missing a positive is costly (disease "
              "screening, fraud). Optimize precision when a false alarm is "
              "costly (spam folder for important email, unnecessary surgery)."},
        {"q": "What does AUC-ROC actually measure, intuitively?",
         "a": "The probability that a randomly chosen positive example is "
              "scored higher than a randomly chosen negative example -- a "
              "threshold-independent measure of ranking quality."},
    ],
    "Calculus & Gradient Descent": [
        {"q": "Why do we subtract the gradient instead of adding it in "
              "gradient descent?",
         "a": "The gradient points in the direction of steepest increase; "
              "subtracting it moves the parameters in the direction of "
              "steepest decrease, i.e. downhill toward lower loss."},
        {"q": "What happens with a learning rate that's too small vs too large?",
         "a": "Too small: convergence is extremely slow and can get stuck in "
              "shallow local structure. Too large: updates overshoot the "
              "minimum, causing oscillation or divergence."},
        {"q": "Why can gradient descent get stuck in a local minimum, and how "
              "do practitioners mitigate this?",
         "a": "On non-convex loss surfaces the gradient is zero at any local "
              "minimum, not just the global one. Momentum, random "
              "restarts/initializations, and stochasticity (mini-batch noise) "
              "all help escape shallow local minima."},
    ],
    "Chain Rule & Backprop": [
        {"q": "Walk through how the chain rule lets you compute dL/dw1 for a "
              "weight in an early layer of a deep network.",
         "a": "You multiply the local derivative at every layer between the "
              "loss and w1: dL/dw1 = dL/dyhat * dyhat/da_last * ... * "
              "da1/dz1 * dz1/dw1, propagating gradients backward one layer "
              "at a time and reusing each intermediate product."},
        {"q": "What is the vanishing gradient problem and how does it relate "
              "to the chain rule?",
         "a": "Because backprop multiplies many local derivatives together, "
              "if each is small (e.g. sigmoid derivatives are at most 0.25), "
              "the product shrinks exponentially with depth, so early layers "
              "receive almost no gradient signal."},
        {"q": "Why is backprop described as 'reverse-mode automatic "
              "differentiation'?",
         "a": "It computes a whole gradient vector (with respect to all "
              "parameters) in a single backward pass by reusing shared "
              "intermediate derivatives, which is far cheaper than "
              "differentiating with respect to each parameter independently "
              "(forward mode) when there are many parameters and one output."},
    ],
}
