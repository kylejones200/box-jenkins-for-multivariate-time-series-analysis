//! AR(1) multi-step forecast.

pub fn ar1_forecast(history: &[f64], phi: f64, intercept: f64, steps: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(steps);
    let mut y = *history.last().unwrap_or(&0.0);
    for _ in 0..steps {
        y = intercept + phi * y;
        out.push(y);
    }
    out
}
