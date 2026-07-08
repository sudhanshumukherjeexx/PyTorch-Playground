# PyTorch Playground ▲

### *learn. loss. repeat.*

An interactive, single-file PyTorch curriculum you run in your browser. No installs, no notebooks, no setup — open one HTML file and start learning PyTorch the way PyTorch learns: **by backpropagating.**

Every lab on the page runs **real math** (forward pass, hand-derived backprop, gradient descent) live in the browser — not animations, not GIFs.

---

## Quick start

```bash
git clone https://github.com/sudhanshumukherjeexx/PyTorch-Playground.git
cd PyTorch-Playground
```

Then just **open `index.html`** in any modern browser (double-click it, or drag it into a tab). That's it — zero dependencies, zero build step, works offline.

Or visit the hosted version on **GitHub Pages**: [sudhanshumukherjeexx.github.io/PyTorch-Playground](https://sudhanshumukherjeexx.github.io/PyTorch-Playground/)

---

## What's inside

- **21 chapters** across four acts, in dependency order.
- **5 live, interactive labs** with real computations.
- **Checkpoint quizzes** at the end of every chapter.
- An **XP + rank system** (Tensor Novice → Grandmaster of Gradients) to track progress in-session.
- A built-in **Reading list** tab — one hand-picked external blog per chapter.

> Progress, XP and quiz scores live in the current session only. The playground stores no data about you.

---

## The live labs

| Lab | Chapter | What it does |
|-----|---------|--------------|
| **Broadcast Bench** | Tensors | Type two tensor shapes and watch the broadcasting rules apply, cell by cell. |
| **The Autograd Tape** | Autograd | A real computational graph for `loss = (w·x + b − t)²` — drag weights, call `backward()`, watch gradients flow, then `step()`. |
| **Training Arena** | NN Fundamentals | A genuine 2-hidden-layer MLP trains in your browser on toy datasets, with a PyTorch code mirror that rewrites itself as you tweak settings. |
| **One Stride at a Time** | CNNs | A 3×3 kernel slides over a 10×10 input — real Sobel / blur / sharpen convolutions, one multiply-accumulate at a time. |
| **Attention, Head by Head** | Transformers | Click any token to see where it attends across different heads. |

---

## Curriculum

**Act I · Foundations**
1. Tensors
2. Neural Network Fundamentals
3. Automatic Differentiation
4. The Training Loop
5. Data Loading & Preprocessing

**Act II · Architectures**
6. Convolutional Networks
7. Recurrent Networks
8. Attention & Transformers
9. Generative Models

**Act III · Production**
10. Model Deployment
11. PyTorch Lightning
12. Distributed Training
13. Custom Extensions
14. Performance Optimization

**Act IV · Frontier**
15. Advanced Architectures
16. Reinforcement Learning
17. Quantization, Pruning & Distillation
18. Meta-Learning
19. Neural Architecture Search
20. Bayesian Deep Learning
21. Research Frontiers

Each chapter pairs opinionated theory and the gotchas that actually bite with runnable code snippets, and — where it helps — one of the live labs.

---

## How it's built

- A **single, self-contained `index.html`** file — HTML, CSS, and vanilla JavaScript, no frameworks or external runtime.
- The only network requests are for Google Fonts; everything else works offline.
- Content lives in a plain data array, rendered by a tiny hash-based router (`#/` for home, `#/ch/<id>` for chapters, `#/blogs` for the reading list).

---

## Archive

The original hands-on Jupyter notebooks (tensor operations, building a neural net from scratch, autograd & fine-tuning, memory tracking & augmentation, and the image / music classification implementations) now live in [`archive/`](archive/), along with their original README.

---

## License

Released under the terms of the [LICENSE](LICENSE) in this repository.

Built by [@sudhanshumukherjeexx](https://github.com/sudhanshumukherjeexx). The playground stays open — go break things. ▲
