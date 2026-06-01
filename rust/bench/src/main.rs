use box_jenkins_for_multivariate_time_series_analysis_core::ar1_forecast;

fn main() {
    let hist: Vec<f64> = (0..120).map(|i| 10.0 + (i as f64 * 0.05).sin()).collect();
    for _ in 0..5000 {
        let _ = ar1_forecast(&hist, 0.7, 0.5, 12);
    }
}
