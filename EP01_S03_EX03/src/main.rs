// file: Episode.01/Section.03/Example.03
//
// Goal: Compare SGD vs Mini-BGD vs BGD for y = w1*x1 + w2*x2 + b using TLS loss (orthogonal distance)
// Note: no external libs; we use a deterministic "rotation" as epoch-wise permutation instead of RNG.

#[derive(Clone, Copy)]
struct Sample {
    x1: f64,
    x2: f64,
    y:  f64,
}

// TLS per-sample loss: 0.5 * r^2 / (w1^2 + w2^2 + 1)
fn loss_per_sample_tls(w1: f64, w2: f64, b: f64, s: Sample) -> f64 {
    let r = w1 * s.x1 + w2 * s.x2 + b - s.y;
    let d = w1 * w1 + w2 * w2 + 1.0;
    0.5 * (r * r) / d
}

// Per-sample gradients (TLS)
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

// Deterministic permutation: simple rotation instead of RNG
fn permute_indices(n: usize, epoch: usize) -> Vec<usize> {
    let shift = epoch % n.max(1);
    let mut v: Vec<usize> = (0..n).collect();
    v.rotate_left(shift);
    v
}

fn main() {
    // Dataset
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

    let lr = 0.0003;
    let epochs = 20_000;
    let n = data.len();

    // 1) SGD
    {
        let (mut w1, mut w2, mut b) = (0.0, 0.0, 0.0);
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let order = permute_indices(n, epoch);

            for &idx in &order {
                let s = data[idx];
                let g1 = grad_w1_tls(w1, w2, b, s);
                let g2 = grad_w2_tls(w1, w2, b, s);
                let gb = grad_b_tls(w1, w2, b, s);
                let l  = loss_per_sample_tls(w1, w2, b, s);
                epoch_loss += l;

                w1 -= lr * g1;
                w2 -= lr * g2;
                b  -= lr * gb;
            }

            if epoch % 1000 == 0 {
                println!("[SGD ] Epoch {:>5}: tls_loss={:.6}, w1={:.6}, w2={:.6}, b={:.6}",
                         epoch, epoch_loss, w1, w2, b);
            }
        }
        println!("\n[SGD ] Final (TLS): w1≈{:.6}, w2≈{:.6}, b≈{:.6}\n", w1, w2, b);
    }

    // 2) Mini-BGD
    {
        let (mut w1, mut w2, mut b) = (0.0, 0.0, 0.0);
        let batch_size = 5;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let order = permute_indices(n, epoch);

            let mut start = 0;
            while start < n {
                let end = (start + batch_size).min(n);
                let batch = &order[start..end];
                let bs = batch.len() as f64;

                let mut g1_sum = 0.0;
                let mut g2_sum = 0.0;
                let mut gb_sum = 0.0;

                for &idx in batch {
                    let s = data[idx];
                    g1_sum += grad_w1_tls(w1, w2, b, s);
                    g2_sum += grad_w2_tls(w1, w2, b, s);
                    gb_sum += grad_b_tls (w1, w2, b, s);
                    epoch_loss += loss_per_sample_tls(w1, w2, b, s);
                }

                let g1 = g1_sum / bs;
                let g2 = g2_sum / bs;
                let gb = gb_sum / bs;

                w1 -= lr * g1;
                w2 -= lr * g2;
                b  -= lr * gb;

                start = end;
            }

            if epoch % 1000 == 0 {
                println!("[Mini] Epoch {:>5}: tls_loss={:.6}, w1={:.6}, w2={:.6}, b={:.6}",
                         epoch, epoch_loss, w1, w2, b);
            }
        }
        println!("\n[Mini] Final (TLS): w1≈{:.6}, w2≈{:.6}, b≈{:.6}\n", w1, w2, b);
    }

    // 3) BGD
    {
        let (mut w1, mut w2, mut b) = (0.0, 0.0, 0.0);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut g1_sum = 0.0;
            let mut g2_sum = 0.0;
            let mut gb_sum = 0.0;

            for s in data.iter().copied() {
                g1_sum += grad_w1_tls(w1, w2, b, s);
                g2_sum += grad_w2_tls(w1, w2, b, s);
                gb_sum += grad_b_tls (w1, w2, b, s);
                epoch_loss += loss_per_sample_tls(w1, w2, b, s);
            }

            let m = n as f64;
            let g1 = g1_sum / m;
            let g2 = g2_sum / m;
            let gb = gb_sum / m;

            w1 -= lr * g1;
            w2 -= lr * g2;
            b  -= lr * gb;

            if epoch % 1000 == 0 {
                println!("[BGD ] Epoch {:>5}: tls_loss={:.6}, w1={:.6}, w2={:.6}, b={:.6}",
                         epoch, epoch_loss, w1, w2, b);
            }
        }
        println!("\n[BGD ] Final (TLS): w1≈{:.6}, w2≈{:.6}, b≈{:.6}\n", w1, w2, b);
    }
}
