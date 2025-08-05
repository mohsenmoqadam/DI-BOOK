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

    let mut w = 0.0;
    let mut b = 0.0;
    let lr = 0.0001;
    let epochs = 10000;

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
