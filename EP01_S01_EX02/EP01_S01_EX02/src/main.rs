// file: Episode.01/Section.02/Example.02
//
// Goal: Demonstrate gradient descent for y = wx + b
// Data: {(1,3), (2,5), (3,7)}

#[derive(Clone, Copy)]
struct Sample {
    x: f64,
    y: f64,
}

// Loss per sample: E = 1/2 * (wx + b - y)^2
fn loss_per_sample(w: f64, b: f64, s: Sample) -> f64 {
    let y_hat = w * s.x + b;
    0.5 * (y_hat - s.y).powi(2)
}

// Gradient w.r.t w: dE/dw = x * (y_hat - y)
fn grad_w(w: f64, b: f64, s: Sample) -> f64 {
    let y_hat = w * s.x + b;
    s.x * (y_hat - s.y)
}

// Gradient w.r.t b: dE/db = (y_hat - y)
fn grad_b(w: f64, b: f64, s: Sample) -> f64 {
    let y_hat = w * s.x + b;
    y_hat - s.y
}

fn main() {
    let data = vec![
        Sample { x: 1.0, y: 3.0 },
        Sample { x: 2.0, y: 5.0 },
        Sample { x: 3.0, y: 7.0 },
    ];

    let mut w = 0.0;
    let mut b = 0.0;
    let lr = 0.1;
    let epochs = 10;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for s in data.iter().copied() {
            let grad_w_val = grad_w(w, b, s);
            let grad_b_val = grad_b(w, b, s);
            let loss = loss_per_sample(w, b, s);
            epoch_loss += loss;

            // Update rule
            w -= lr * grad_w_val;
            b -= lr * grad_b_val;
        }

        println!(
            "Epoch {:>2}: loss={:.6}, w={:.6}, b={:.6}",
            epoch, epoch_loss, w, b
        );
    }

    println!("\nFinal parameters: w ≈ {:.6}, b ≈ {:.6}", w, b);
}
