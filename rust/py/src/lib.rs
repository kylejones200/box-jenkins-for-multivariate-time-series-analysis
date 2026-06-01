use box_jenkins_for_multivariate_time_series_analysis_core::ar1_forecast;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn ar1_forecast_py<'py>(py: Python<'py>, history: PyReadonlyArray1<f64>, phi: f64, intercept: f64, steps: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(ar1_forecast(history.as_slice()?, phi, intercept, steps).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (history, phi, intercept, steps, iterations=500))]
fn bench_kernel_py(history: PyReadonlyArray1<f64>, phi: f64, intercept: f64, steps: usize, iterations: usize) -> PyResult<f64> {
    let history_buf = history.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = ar1_forecast(&history_buf, phi, intercept, steps);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn box_jenkins_for_multivariate_time_series_analysis_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ar1_forecast_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
