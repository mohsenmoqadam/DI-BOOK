# 📘 DI-BOOK – Episode 01 / Section 02 / Example 01

This repository contains the source code and exercises for the book **"Digital Intelligence (DI-BOOK)"**, designed for software engineers who want to deeply understand how modern AI works – step by step, from scratch.

---

## 🧠 Example: Two-Parameter Linear Fit

In this example, we implement a simple but complete linear model with two trainable parameters:

\[
y = w \cdot x + b
\]

Unlike the previous example (which assumed the line passes through the origin), this model can learn a vertical shift (bias term).

---

## 🧪 Dataset

We use a tiny, synthetic dataset of three samples:
‍‍‍```S = {(1, 3), (2, 5), (3, 7)}```

This dataset is perfectly aligned with the line \( y = 2x + 1 \), which the model should learn if gradient descent is implemented correctly.

---

## ⚙️ Objective

- Demonstrate **manual gradient computation** for both parameters `w` and `b`
- Implement basic **gradient descent** over multiple epochs
- Track and print parameter updates and total loss per epoch

---

## 🚀 How to Run

Follow the steps below to run the example:

1. **Clone the repository:**
```bash
git clone https://github.com/mohsenmoqadam/DI-BOOK.git
```
3. **Navigate to the example directory:**
```bash
cd DI-BOOK/EP01_S02_EX01/
```
5. **Run the code using Cargo:**
```bash
cargo run
```
5. **Output:**
  ```bash
  ❯ cargo run
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.00s
     Running `target/debug/EP01_S01_EX02`
Epoch  0: loss=17.197450, w=1.999000, b=1.003000
Epoch  1: loss=0.000003, w=1.999044, b=1.002868
Epoch  2: loss=0.000002, w=1.999086, b=1.002742
Epoch  3: loss=0.000002, w=1.999126, b=1.002621
Epoch  4: loss=0.000002, w=1.999165, b=1.002506
Epoch  5: loss=0.000002, w=1.999201, b=1.002396
Epoch  6: loss=0.000002, w=1.999237, b=1.002290
Epoch  7: loss=0.000002, w=1.999270, b=1.002189
Epoch  8: loss=0.000001, w=1.999302, b=1.002093
Epoch  9: loss=0.000001, w=1.999333, b=1.002001
Epoch 10: loss=0.000001, w=1.999362, b=1.001913
Epoch 11: loss=0.000001, w=1.999390, b=1.001829
Epoch 12: loss=0.000001, w=1.999417, b=1.001748
Epoch 13: loss=0.000001, w=1.999443, b=1.001671
Epoch 14: loss=0.000001, w=1.999467, b=1.001598
Epoch 15: loss=0.000001, w=1.999491, b=1.001528
Epoch 16: loss=0.000001, w=1.999513, b=1.001460
Epoch 17: loss=0.000001, w=1.999535, b=1.001396
Epoch 18: loss=0.000001, w=1.999555, b=1.001335
Epoch 19: loss=0.000001, w=1.999575, b=1.001276
Epoch 20: loss=0.000000, w=1.999593, b=1.001220

Final parameters: w ≈ 1.999593, b ≈ 1.001220
  ```
