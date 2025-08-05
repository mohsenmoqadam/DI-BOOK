# 📘 DI-BOOK – Episode 01 / Section 02 / Example 02

This repository contains the source code and exercises for the book **"Digital Intelligence (DI-BOOK)"**, designed for software engineers who want to deeply understand how modern AI works – step by step, from scratch.

---

## 🧠 Example: Fitting a Line to Noisy Data

In this example, we use a linear model with two trainable parameters:

**Model:** *y = w·x + b*

Unlike earlier examples where all data points perfectly aligned on a straight line, here we work with noisy, real-world-like data. The model is expected to find the best-fit line that minimizes total error across all samples.

---

## 🧪 Dataset

We use a synthetic dataset of 20 points, generated around the underlying line ‍`( y = 3x )`, with some added noise:

```rust
let data = vec![
    Sample { x: 1.0, y: 3.9934 },
    Sample { x: 1.4737, y: 4.1445 },
    Sample { x: 1.9474, y: 7.1375 },
    Sample { x: 2.4211, y: 10.3092 },
    Sample { x: 2.8947, y: 8.2159 },
    Sample { x: 3.3684, y: 8.8467 },
    Sample { x: 3.8421, y: 11.7036 },
    Sample { x: 4.3158, y: 11.3982 },
    Sample { x: 4.7895, y: 14.1913 },
    Sample { x: 5.2632, y: 13.0675 },
    Sample { x: 5.7368, y: 17.4839 },
    Sample { x: 6.2105, y: 18.7910 },
    Sample { x: 6.6842, y: 19.2922 },
    Sample { x: 7.1579, y: 20.4346 },
    Sample { x: 7.6316, y: 20.5611 },
    Sample { x: 8.1053, y: 23.4436 },
    Sample { x: 8.5789, y: 23.0844 },
    Sample { x: 9.0526, y: 26.2921 },
    Sample { x: 9.5263, y: 26.8375 },
    Sample { x: 10.0, y: 30.0855 },
];
```
## ⚙️ Objective

- Show how gradient descent works with imperfect, noisy data
- Learn a best-fit line that minimizes total error
- Observe convergence of parameters w and b toward a statistically optimal solution

---

## 🚀 How to Run

Follow the steps below to run the example:

1. **Clone the repository:**
```bash
git clone https://github.com/mohsenmoqadam/DI-BOOK.git
```
3. **Navigate to the example directory:**
```bash
cd DI-BOOK/EP01_S02_EX02/
```
5. **Run the code using Cargo:**
```bash
cargo run
```
5. **Output:**
  ```bash
  ❯ cargo run
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.00s
     Running `target/debug/EP01_S02_EX02`
Epoch  0: loss=2912.774735, w=0.209030, b=0.031051
Epoch  1: loss=2497.568007, w=0.402514, b=0.059806
Epoch  2: loss=2141.826169, w=0.581606, b=0.086436
Epoch  3: loss=1837.033129, w=0.747377, b=0.111099
Epoch  4: loss=1575.892378, w=0.900818, b=0.133941
Epoch  5: loss=1352.152335, w=1.042847, b=0.155099
Epoch  6: loss=1160.456708, w=1.174310, b=0.174696
Epoch  7: loss=996.216279, w=1.295995, b=0.192849
Epoch  8: loss=855.499053, w=1.408629, b=0.209665
Epoch  9: loss=734.936141, w=1.512884, b=0.225245
Epoch 10: loss=631.641114, w=1.609384, b=0.239679
Epoch 11: loss=543.140913, w=1.698706, b=0.253053
Epoch 12: loss=467.316652, w=1.781383, b=0.265446
Epoch 13: loss=402.352893, w=1.857909, b=0.276931
Epoch 14: loss=346.694194, w=1.928742, b=0.287575
Epoch 15: loss=299.007873, w=1.994306, b=0.297441
Epoch 16: loss=258.152107, w=2.054992, b=0.306586
Epoch 17: loss=223.148596, w=2.111162, b=0.315065
Epoch 18: loss=193.159152, w=2.163154, b=0.322927
Epoch 19: loss=167.465624, w=2.211277, b=0.330217
Epoch 20: loss=145.452717, w=2.255819, b=0.336979
...
Epoch 9983: loss=13.418106, w=2.744205, b=0.875113
Epoch 9984: loss=13.418105, w=2.744205, b=0.875116
Epoch 9985: loss=13.418105, w=2.744204, b=0.875120
Epoch 9986: loss=13.418105, w=2.744204, b=0.875124
Epoch 9987: loss=13.418105, w=2.744203, b=0.875128
Epoch 9988: loss=13.418104, w=2.744202, b=0.875131
Epoch 9989: loss=13.418104, w=2.744202, b=0.875135
Epoch 9990: loss=13.418104, w=2.744201, b=0.875139
Epoch 9991: loss=13.418104, w=2.744201, b=0.875143
Epoch 9992: loss=13.418104, w=2.744200, b=0.875147
Epoch 9993: loss=13.418103, w=2.744200, b=0.875150
Epoch 9994: loss=13.418103, w=2.744199, b=0.875154
Epoch 9995: loss=13.418103, w=2.744199, b=0.875158
Epoch 9996: loss=13.418103, w=2.744198, b=0.875162
Epoch 9997: loss=13.418103, w=2.744197, b=0.875165
Epoch 9998: loss=13.418102, w=2.744197, b=0.875169
Epoch 9999: loss=13.418102, w=2.744196, b=0.875173

Final parameters: w ≈ 2.744196, b ≈ 0.875173
  ```
