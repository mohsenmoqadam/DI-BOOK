// file: Episode.01/Section.01/Example.01
//
// Goal: Demonstrate explicit gradient calculation and w update
// Model: y = w * x
// Data: {(1,2), (2,4), (3,6)}

#[derive(Clone, Copy)]
struct Sample {
    x: f64,
    y: f64,
}

// Loss per sample: E = 1/2 * (w * x - y)^2
fn loss_per_sample(w: f64, s: Sample) -> f64 {
    let y_hat = w * s.x;
    0.5 * (y_hat - s.y).powi(2)
}

// Gradient of the loss w.r.t. w: dE/dw = x * (w * x - y)
fn grad_per_sample(w: f64, s: Sample) -> f64 {
    let y_hat = w * s.x;
    s.x * (y_hat - s.y)
}

fn main() {
    // Example dataset
    let data = vec![
        Sample { x: 1.0, y: 2.0 },
        Sample { x: 2.0, y: 4.0 },
        Sample { x: 3.0, y: 6.0 },
    ];

    // Initial guess for w (intentionally poor)
    let mut w: f64 = 0.0;

    // Learning rate (step size)
    let lr: f64 = 0.1;

    // Number of epochs (passes over the full dataset)
    let epochs: usize = 5;

    for epoch in 0..epochs {
        println!("================ EPOCH {epoch} ================");
        let mut epoch_loss = 0.0;

        for (i, s) in data.iter().copied().enumerate() {
            let y_hat = w * s.x;
            let error = y_hat - s.y;
            let grad = grad_per_sample(w, s);
            let loss = loss_per_sample(w, s);
            epoch_loss += loss;

            println!(
                "[sample #{:>2}] x={:.1}, y={:.1} | y_hat={:.6}, error={:.6}, grad(dE/dw)={:.6}, loss={:.6}, w(before)={:.6}",
                i, s.x, s.y, y_hat, error, grad, loss, w
            );

            // Update w using gradient descent
            w -= lr * grad;

            println!(
                "             --> w(after)={:.6}\n",
                w
            );
        }

        println!("epoch_loss = {:.12}\n", epoch_loss);
    }
    println!("FINAL w ≈ {:.12}", w);
}
