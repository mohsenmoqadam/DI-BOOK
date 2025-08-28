// file: Episode.01/Section.03/Example.01
//
// Goal: Gradient Descent for y = w1*x1 + w2*x2 + b (OLS loss)
// Data: small synthetic dataset (no external libs)

#[derive(Clone, Copy)]
struct Sample {
    x1: f64,
    x2: f64,
    y:  f64,
}

// Loss per sample (OLS): E = 1/2 * (y_hat - y)^2
fn loss_per_sample(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let y_hat = w1 * s.x1 + w2 * s.x2 + b;
    0.5 * (y_hat - s.y).powi(2)
}

// Gradients:
// dE/dw1 = x1 * (y_hat - y)
// dE/dw2 = x2 * (y_hat - y)
// dE/db  =       (y_hat - y)
fn grad_w1(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let y_hat = w1 * s.x1 + w2 * s.x2 + b;
    s.x1 * (y_hat - s.y)
}
fn grad_w2(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let y_hat = w1 * s.x1 + w2 * s.x2 + b;
    s.x2 * (y_hat - s.y)
}
fn grad_b(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let y_hat = w1 * s.x1 + w2 * s.x2 + b;
    y_hat - s.y
}

fn main() {
    // Synthetic data generated from approx: y ≈ 2.5*x1 - 1.2*x2 + 0.8 + noise
    let data = vec![
        Sample { x1: 1.25,  x2: 6.50, y: -3.6683 },
        Sample { x1: 1.70,  x2: 6.20, y: -2.2352 },
        Sample { x1: 2.15,  x2: 5.90, y: -0.9527 },
        Sample { x1: 2.60,  x2: 5.60, y:  0.4354 },
        Sample { x1: 3.05,  x2: 5.30, y:  2.0718 },
        Sample { x1: 3.50,  x2: 5.00, y:  3.0465 },
        Sample { x1: 3.95,  x2: 4.70, y:  4.7427 },
        Sample { x1: 4.40,  x2: 4.40, y:  5.4380 },
        Sample { x1: 4.85,  x2: 4.10, y:  7.0798 },
        Sample { x1: 5.30,  x2: 3.80, y:  8.3566 },
        Sample { x1: 5.75,  x2: 3.50, y: 10.0216 },
        Sample { x1: 6.20,  x2: 3.20, y: 11.1328 },
        Sample { x1: 6.65,  x2: 2.90, y: 12.4977 },
        Sample { x1: 7.10,  x2: 2.60, y: 14.0914 },
        Sample { x1: 7.55,  x2: 2.30, y: 15.1160 },
        Sample { x1: 8.00,  x2: 2.00, y: 18.2503 },
        Sample { x1: 8.45,  x2: 1.70, y: 20.1308 },
        Sample { x1: 8.90,  x2: 1.40, y: 21.6597 },
        Sample { x1: 9.35,  x2: 1.10, y: 23.0411 },
        Sample { x1: 9.80,  x2: 0.80, y: 24.5813 },
    ];

    // Params
    let mut w1 = 0.0;
    let mut w2 = 0.0;
    let mut b  = 0.0;

    let lr = 0.0005;    // learning rate (η)
    let epochs = 20000; // iterations

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        // SGD-style: update per sample
        for s in data.iter().copied() {
            let g1 = grad_w1(w1, w2, b, s);
            let g2 = grad_w2(w1, w2, b, s);
            let gb = grad_b(w1, w2, b, s);
            let l  = loss_per_sample(w1, w2, b, s);
            epoch_loss += l;

            // update
            w1 -= lr * g1;
            w2 -= lr * g2;
            b  -= lr * gb;
        }

        if epoch % 1000 == 0 {
            println!(
                "Epoch {:>5}: loss={:.6}, w1={:.6}, w2={:.6}, b={:.6}",
                epoch, epoch_loss, w1, w2, b
            );
        }
    }

    println!("\nFinal parameters: w1 ≈ {:.6}, w2 ≈ {:.6}, b ≈ {:.6}", w1, w2, b);
}
