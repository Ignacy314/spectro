use plotly::{Plot, Scatter, common::Mode};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

use circular_buffer::CircularBuffer;
use ndarray::Array2;
use smartcore::{
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    metrics::{Metrics, RegressionMetrics},
    model_selection::train_test_split,
};

use crate::process_samples;

pub fn train_model<P: AsRef<Path>>(
    wav_path: P,
    csv_path: P,
    out_path: P,
) -> RandomForestRegressor<f32, f32, Array2<f32>, Vec<f32>> {
    let mut wav = hound::WavReader::open(wav_path).unwrap();
    let mut buffer: CircularBuffer<8192, i32> = CircularBuffer::new();
    let mut counter = 0;
    let mut x_data = Vec::new();
    let mut row_len = 0;
    for s in wav.samples::<i32>() {
        let sample = s.unwrap();
        buffer.push_back(sample);
        counter += 1;
        if buffer.is_full() && counter >= 2400 {
            counter = 0;
            let (_freqs, values) = process_samples(buffer.iter());
            if row_len != 0 {
                assert_eq!(row_len, values.len());
            }
            row_len = values.len();
            x_data.extend(values);
        }
    }

    let mut y_reader = csv::Reader::from_path(csv_path).unwrap();
    let mut y_data = Vec::new();
    for result in y_reader.deserialize() {
        let (_, dist): (f32, f32) = result.unwrap();
        y_data.push(dist);
    }

    let n_y = y_data.len();
    let mut n_x = x_data.len() / row_len;
    match n_y.cmp(&n_x) {
        std::cmp::Ordering::Less => {
            x_data.truncate(n_y * row_len);
            n_x = n_y;
        }
        std::cmp::Ordering::Greater => {
            y_data.truncate(n_x);
        }
        _ => {}
    }

    let x = Array2::from_shape_vec((n_x, row_len), x_data).unwrap();
    let y = y_data;

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(42));

    let model = RandomForestRegressor::fit(
        &x_train,
        &y_train,
        RandomForestRegressorParameters::default().with_seed(42),
    )
    .unwrap();

    let y_hat = model.predict(&x_test).unwrap();

    let mse = RegressionMetrics::mean_squared_error().get_score(&y_test, &y_hat);
    let r2 = RegressionMetrics::r2().get_score(&y_test, &y_hat);
    println!("MSE: {mse} | R2: {r2}");

    let x: Vec<usize> = (0..y_test.len()).collect();
    let mut plot = Plot::new();
    let y_test_plot = Scatter::new(x.clone(), y_test);
    let y_hat_plot = Scatter::new(x, y_hat).mode(Mode::Markers);
    plot.add_traces(vec![y_test_plot, y_hat_plot]);
    plot.write_html(out_path.as_ref().join("html"));

    let model_bytes = bincode::serialize(&model).unwrap();
    File::create(out_path)
        .and_then(|mut f| f.write_all(&model_bytes))
        .expect("Can not persist the model");

    model
}

pub fn load_model<P: AsRef<Path>>(
    model_path: P,
) -> RandomForestRegressor<f32, i32, Array2<f32>, Vec<i32>> {
    bincode::deserialize_from(BufReader::new(File::open(model_path).unwrap())).unwrap()
}
