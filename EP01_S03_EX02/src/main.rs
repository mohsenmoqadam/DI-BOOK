// file: Episode.01/Section.03/Example.02
//
// Goal: SGD for y = w1*x1 + w2*x2 + b using TLS loss (orthogonal distance)
// Data: same synthetic dataset (no external libs)

#[derive(Clone, Copy)]
struct Sample {
    x1: f64,
    x2: f64,
    y:  f64,
}

// TLS per-sample loss: 0.5 * r^2 / (w1^2 + w2^2 + 1)
// where r = (w1*x1 + w2*x2 + b - y)
fn loss_per_sample_tls(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let r = w1 * s.x1 + w2 * s.x2 + b - s.y;
    let d = w1 * w1 + w2 * w2 + 1.0;
    0.5 * (r * r) / d
}

// Gradients for TLS per sample:
// dL/dw1 = (r*x1)/D - (r^2*w1)/D^2
// dL/dw2 = (r*x2)/D - (r^2*w2)/D^2
// dL/db  =  r/D
fn grad_w1_tls(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let r = w1 * s.x1 + w2 * s.x2 + b - s.y;
    let d = w1 * w1 + w2 * w2 + 1.0;
    (r * s.x1) / d - (r * r * w1) / (d * d)
}
fn grad_w2_tls(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let r = w1 * s.x1 + w2 * s.x2 + b - s.y;
    let d = w1 * w1 + w2 * w2 + 1.0;
    (r * s.x2) / d - (r * r * w2) / (d * d)
}
fn grad_b_tls(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let r = w1 * s.x1 + w2 * s.x2 + b - s.y;
    let d = w1 * w1 + w2 * w2 + 1.0;
    r / d
}

fn main() {
    // Same dataset
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

    // TLS gradients can be more sensitive; start smaller than OLS
    let lr = 0.0003;    // learning rate (η)
    let epochs = 20000; // iterations

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        // SGD-style: update per sample
        for s in data.iter().copied() {
            let g1 = grad_w1_tls(w1, w2, b, s);
            let g2 = grad_w2_tls(w1, w2, b, s);
            let gb = grad_b_tls(w1, w2, b, s);
            let l  = loss_per_sample_tls(w1, w2, b, s);
            epoch_loss += l;

            // update
            w1 -= lr * g1;
            w2 -= lr * g2;
            b  -= lr * gb;
        }

        if epoch % 1000 == 0 {
            println!(
                "Epoch {:>5}: tls_loss={:.6}, w1={:.6}, w2={:.6}, b={:.6}",
                epoch, epoch_loss, w1, w2, b
            );
        }
    }

    println!("\nFinal (TLS) parameters: w1 ≈ {:.6}, w2 ≈ {:.6}, b ≈ {:.6}", w1, w2, b);
}
