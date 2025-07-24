# DI-BOOK – Episode 01 / Section 01 / Example 01

This repository contains the source code and exercises for the book "Digital Intelligence (DI-BOOK)", written for software engineers who want to deeply understand how modern AI works – step by step, from scratch.

## 📘 Example: Single-Parameter Linear Fit

In this example, we implement a simple linear model:  
**y = w * x**  
The goal is to demonstrate how error gradients are calculated and how the weight `w` is updated iteratively using gradient descent.

---

## 🚀 How to Run

Follow the steps below to run the example:

1. **Clone the repository:**
```bash
git clone https://github.com/mohsenmoqadam/DI-BOOK.git
```
3. **Navigate to the example directory:**
```bash
cd DI-BOOK/EP01_S01_EX01/
```
5. **Run the code using Cargo:**
```bash
cargo run
```
5. **Output:**
  ```bash
  ❯ cargo run
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.03s
     Running `target/debug/EP01_S01_EX01`
================ EPOCH 0 ================
[sample # 0] x=1.0, y=2.0 | y_hat=0.000000, error=-2.000000, grad(dE/dw)=-2.000000, loss=2.000000, w(before)=0.000000
             --> w(after)=0.200000

[sample # 1] x=2.0, y=4.0 | y_hat=0.400000, error=-3.600000, grad(dE/dw)=-7.200000, loss=6.480000, w(before)=0.200000
             --> w(after)=0.920000

[sample # 2] x=3.0, y=6.0 | y_hat=2.760000, error=-3.240000, grad(dE/dw)=-9.720000, loss=5.248800, w(before)=0.920000
             --> w(after)=1.892000

epoch_loss = 13.728800000000

================ EPOCH 1 ================
[sample # 0] x=1.0, y=2.0 | y_hat=1.892000, error=-0.108000, grad(dE/dw)=-0.108000, loss=0.005832, w(before)=1.892000
             --> w(after)=1.902800

[sample # 1] x=2.0, y=4.0 | y_hat=3.805600, error=-0.194400, grad(dE/dw)=-0.388800, loss=0.018896, w(before)=1.902800
             --> w(after)=1.941680

[sample # 2] x=3.0, y=6.0 | y_hat=5.825040, error=-0.174960, grad(dE/dw)=-0.524880, loss=0.015306, w(before)=1.941680
             --> w(after)=1.994168

epoch_loss = 0.040033180800

================ EPOCH 2 ================
[sample # 0] x=1.0, y=2.0 | y_hat=1.994168, error=-0.005832, grad(dE/dw)=-0.005832, loss=0.000017, w(before)=1.994168
             --> w(after)=1.994751

[sample # 1] x=2.0, y=4.0 | y_hat=3.989502, error=-0.010498, grad(dE/dw)=-0.020995, loss=0.000055, w(before)=1.994751
             --> w(after)=1.996851

[sample # 2] x=3.0, y=6.0 | y_hat=5.990552, error=-0.009448, grad(dE/dw)=-0.028344, loss=0.000045, w(before)=1.996851
             --> w(after)=1.999685

epoch_loss = 0.000116736755

================ EPOCH 3 ================
[sample # 0] x=1.0, y=2.0 | y_hat=1.999685, error=-0.000315, grad(dE/dw)=-0.000315, loss=0.000000, w(before)=1.999685
             --> w(after)=1.999717

[sample # 1] x=2.0, y=4.0 | y_hat=3.999433, error=-0.000567, grad(dE/dw)=-0.001134, loss=0.000000, w(before)=1.999717
             --> w(after)=1.999830

[sample # 2] x=3.0, y=6.0 | y_hat=5.999490, error=-0.000510, grad(dE/dw)=-0.001531, loss=0.000000, w(before)=1.999830
             --> w(after)=1.999983

epoch_loss = 0.000000340404

================ EPOCH 4 ================
[sample # 0] x=1.0, y=2.0 | y_hat=1.999983, error=-0.000017, grad(dE/dw)=-0.000017, loss=0.000000, w(before)=1.999983
             --> w(after)=1.999985

[sample # 1] x=2.0, y=4.0 | y_hat=3.999969, error=-0.000031, grad(dE/dw)=-0.000061, loss=0.000000, w(before)=1.999985
             --> w(after)=1.999991

[sample # 2] x=3.0, y=6.0 | y_hat=5.999972, error=-0.000028, grad(dE/dw)=-0.000083, loss=0.000000, w(before)=1.999991
             --> w(after)=1.999999

epoch_loss = 0.000000000993

FINAL w ≈ 1.999999081670
  ```
