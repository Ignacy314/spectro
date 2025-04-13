use ndarray::{Array, Array2, ArrayViewD};
use ort::inputs;
use plotly::{Plot, Scatter, common::Mode};
use regex::Regex;
use serde::Deserialize;
use std::{
    fs::File,
    io::BufWriter,
    io::Write,
    path::{Path, PathBuf},
};

use circular_buffer::CircularBuffer;

use crate::{load_onnx, process_samples};

#[allow(unused)]
#[derive(Deserialize)]
struct Record {
    clock: f32,
    lon: f64,
    lat: f64,
    alt: f64,
    rfc: String,
    distance: f64,
}

fn read_data<P: AsRef<Path>>(input_dir: P, module: i32) -> (Array2<f32>, Vec<f64>) {
    println!("reading data for module {module}");

    let module_str = module.to_string();

    let flights = std::fs::read_dir(input_dir.as_ref().join("umc")).unwrap();
    let mut flights_wavs: Vec<PathBuf> = flights
        .map(|f| f.unwrap().path().join(&module_str))
        .flat_map(|p| std::fs::read_dir(p).unwrap().map(|d| d.unwrap().path()))
        .collect();
    let mut flights_csvs: Vec<PathBuf> =
        std::fs::read_dir(input_dir.as_ref().join("module_csvs").join(&module_str))
            .unwrap()
            .map(|d| d.unwrap().path())
            .collect();

    assert_eq!(flights_wavs.len(), flights_csvs.len());

    let re = Regex::new(r".*(\d+)\.wav$").unwrap();
    flights_wavs.sort_unstable_by(|a, b| {
        let a_num: i32 = re.captures(a.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        let b_num: i32 = re.captures(b.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        a_num.cmp(&b_num)
    });

    let re = Regex::new(r".*(\d+)\.csv$").unwrap();
    flights_csvs.sort_unstable_by(|a, b| {
        let a_num: i32 = re.captures(a.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        let b_num: i32 = re.captures(b.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        a_num.cmp(&b_num)
    });

    // eprintln!("{flights_csvs:?}");
    // eprintln!("{flights_wavs:?}");

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut row_len = 0;
    for (wav_path, csv_path) in flights_wavs.iter().zip(flights_csvs.iter()) {
        // eprintln!("{wav_path:?} | {csv_path:?}");
        let mut buffer: CircularBuffer<8192, i32> = CircularBuffer::new();
        let mut counter = 0;
        let mut wav = hound::WavReader::open(wav_path).unwrap();
        let mut csv = csv::Reader::from_path(csv_path).unwrap();

        let mut distances = Vec::new();
        for result in csv.deserialize() {
            let r: Record = result.unwrap();
            distances.push(r.distance);
        }

        // to test if csv size and wav length more or less match
        // let n_csv_records = distances.len();
        // let n_wav_periods = wav.duration() / 2400;
        // eprintln!("{n_csv_records} {n_wav_periods}");

        let mut dist_iter = distances.iter();

        for s in wav.samples::<i32>() {
            let sample = s.unwrap();
            buffer.push_back(sample);
            counter += 1;
            if buffer.is_full() && counter >= 2400 {
                let Some(distance) = dist_iter.next() else {
                    break;
                };
                counter = 0;
                let (_freqs, values) = process_samples(buffer.iter());
                if row_len != 0 {
                    assert_eq!(row_len, values.len());
                }
                row_len = values.len();
                x_data.extend(values);
                y_data.push(*distance);
            }
        }
    }

    let n_y = y_data.len();
    let mut n = x_data.len() / row_len;
    match n_y.cmp(&n) {
        std::cmp::Ordering::Less => {
            x_data.truncate(n_y * row_len);
            n = n_y;
        }
        std::cmp::Ordering::Greater => {
            y_data.truncate(n);
        }
        _ => {}
    }

    let x = Array2::from_shape_vec((n, row_len), x_data).unwrap();
    let y = y_data;

    (x, y)
}

pub fn generate_data_csv<P: AsRef<Path>>(
    input_dir: P,
    module: i32,
    out_path: P,
    _bad_flights: Vec<i32>,
) {
    let (x, y) = read_data(input_dir, module);
    let mut csv = BufWriter::new(File::create(out_path).unwrap());
    for (y, xs) in y.iter().zip(x.outer_iter()) {
        let n_xs = xs.len();
        write!(csv, "{y},").unwrap();
        let x_vec = xs.to_vec();
        for x in &x_vec[0..(n_xs - 1)] {
            write!(csv, "{x},").unwrap();
        }
        writeln!(csv, "{}", x_vec[n_xs - 1]).unwrap();
    }
}

fn read_data_csv<P: AsRef<Path>>(csv_path: P) -> (Vec<Vec<f32>>, Vec<f64>) {
    let mut csv = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(csv_path)
        .unwrap();

    let mut x = Vec::new();
    let mut y = Vec::new();

    for result in csv.deserialize() {
        let (y_data, x_data): (f64, Vec<f32>) = result.unwrap();
        y.push(y_data);
        x.push(x_data);
    }

    (x, y)
}

// pub fn train_model<P: AsRef<Path>>(
//     input_dir: P,
//     module: i32,
//     out_path: P,
// ) -> RandomForestRegressor<f32, f64, Array2<f32>, Vec<f64>> {
//     let (x, y) = read_data(input_dir, module);
//
//     let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, false, Some(42));
//
//     println!("training module {module}");
//     let model = RandomForestRegressor::fit(
//         &x_train,
//         &y_train,
//         RandomForestRegressorParameters::default()
//             .with_seed(42)
//             .with_n_trees(32),
//     )
//     .unwrap();
//
//     println!("metrics");
//     let y_hat = model.predict(&x_test).unwrap();
//
//     let mse = RegressionMetrics::mean_squared_error().get_score(&y_test, &y_hat);
//     let r2 = RegressionMetrics::r2().get_score(&y_test, &y_hat);
//     println!("MSE: {mse} | R2: {r2}");
//
//     let x: Vec<usize> = (0..y_test.len()).collect();
//     let mut plot = Plot::new();
//     let y_test_plot = Scatter::new(x.clone(), y_test);
//     let y_hat_plot = Scatter::new(x, y_hat).mode(Mode::Markers);
//     plot.add_traces(vec![y_hat_plot, y_test_plot]);
//     plot.write_html(out_path.as_ref().with_extension("html"));
//
//     let model_bytes = bincode::serialize(&model).unwrap();
//     File::create(out_path)
//         .and_then(|mut f| f.write_all(&model_bytes))
//         .expect("Can not persist the model");
//
//     model
// }

// pub fn load_model<P: AsRef<Path>>(
//     model_path: P,
// ) -> RandomForestRegressor<f32, f64, Array2<f32>, Vec<f64>> {
//     bincode::deserialize_from(BufReader::new(File::open(model_path).unwrap())).unwrap()
// }

pub fn test_onnx<P: AsRef<Path>>(model_path: P, input_csv: P, plot_path: P) {
    println!("loading onnx model");
    let model = load_onnx(model_path);

    println!("reading data");
    let (x, y) = read_data_csv(input_csv);
    let x_shape = (x.len(), x[0].len());
    let x = Array::from_iter(x.into_iter().flatten())
        .into_shape_with_order(x_shape)
        .unwrap();

    println!("testing onnx model");

    let outputs = model.run(inputs![x].unwrap()).unwrap();

    let y_pred: ArrayViewD<f64> = outputs["variable"].try_extract_tensor().unwrap();
    let n_y_pred = y_pred.len();
    let y_pred = y_pred.into_shape_with_order(n_y_pred).unwrap();
    println!("number of outputs: {}", y_pred.len());

    let y_avg: Vec<f64> = y
        .windows(20)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect();
    let y_pred_avg: Vec<f64> = y_pred
        .to_vec()
        .windows(20)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect();

    let x: Vec<usize> = (0..y_avg.len()).collect();
    let mut plot = Plot::new();
    let y_test_plot = Scatter::new(x.clone(), y_avg);
    let y_hat_plot = Scatter::new(x, y_pred_avg).mode(Mode::Markers);
    plot.add_traces(vec![y_hat_plot, y_test_plot]);
    plot.write_html(plot_path);

    // let x_tract =
    //     tract_ndarray::Array2::from_shape_vec((x_shape[0], x_shape[1]), x.into_raw_vec()).unwrap();
    // let x_tensor: Tensor = x_tract.into();
    //
    // let result = model.run(tvec!(x_tensor.into())).unwrap();
    //
    // println!("{:?}", result[0]);
}

// pub fn test_avg<P: AsRef<Path>>(model_path: P, input_dir: P, module: i32, plot_path: P) {
//     let model = load_model(model_path);
//
//     let (x, y) = read_data(input_dir, module);
//
//     let y_hat = model.predict(&x).unwrap();
//
//     let y_avg: Vec<f64> = y.windows(20).map(|w| w.sum() / w.len() as f64).collect();
//     let y_hat_avg: Vec<f64> = y_hat
//         .windows(20)
//         .map(|w| w.sum() / w.len() as f64)
//         .collect();
//
//     let x: Vec<usize> = (0..y_avg.len()).collect();
//     let mut plot = Plot::new();
//     let y_test_plot = Scatter::new(x.clone(), y_avg);
//     let y_hat_plot = Scatter::new(x, y_hat_avg).mode(Mode::Markers);
//     plot.add_traces(vec![y_hat_plot, y_test_plot]);
//     plot.write_html(plot_path);
// }
