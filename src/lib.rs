use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use convolutions_rs::convolutions::*;
// use interp::{InterpMode, interp_slice};
use ndarray::*;
use smartcore::{
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::basic::matrix::DenseMatrix,
    metrics::{ClassificationMetricsOrd, Metrics},
    model_selection::train_test_split,
    tree::decision_tree_classifier::SplitCriterion,
};
use spectrum_analyzer::{samples_fft_to_spectrum, windows::hann_window};

pub fn process_samples(samples: &[i32]) -> (Vec<f32>, Vec<f32>) {
    let samples = samples.iter().map(|s| *s as f32).collect::<Vec<_>>();
    let hann_window = hann_window(&samples);

    let spectrum = samples_fft_to_spectrum(
        &hann_window,
        48000,
        spectrum_analyzer::FrequencyLimit::Range(5.0, 4000.0),
        // spectrum_analyzer::FrequencyLimit::All,
        None,
    )
    .unwrap();

    // let frequencies: Vec<f32> = (5..=4000).map(|s| s as f32).collect();

    let (freqs, values): (Vec<_>, Vec<_>) = spectrum.data().iter().copied().unzip();
    let freqs: Vec<f32> = freqs.into_iter().map(|f| f.val()).collect();

    let values: Vec<f32> = values.iter().map(|s| s.val().abs()).collect();
    let input = Array::from_shape_vec((1, 1, values.len()), values.clone()).unwrap();
    let kernel: Array4<f32> = Array::from_shape_vec((1, 1, 1, 21), vec![1.0 / 21.0; 21]).unwrap();
    let conv_layer = ConvolutionLayer::new(kernel, None, 1, convolutions_rs::Padding::Same);
    let output_layer: Array3<f32> = conv_layer.convolve(&input);
    let output_layer = output_layer.into_raw_vec();

    let mut fft_diff = values
        .iter()
        .zip(output_layer.iter())
        .map(|(v, a)| v - a)
        .collect::<Vec<f32>>();
    let min_diff = *fft_diff.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_diff = *fft_diff.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    if max_diff > min_diff {
        fft_diff
            .iter_mut()
            .for_each(|s| *s = 2.0 * (*s - min_diff) / (max_diff - min_diff) - 1.0);
    } else {
        fft_diff = vec![0.0; fft_diff.len()];
    }

    // let interp_fft_diff = interp_slice(&freqs, &fft_diff, &frequencies, &InterpMode::default());
    // assert_eq!(interp_fft_diff.len(), 3996);

    (freqs, fft_diff)
}

pub fn wav_to_csv<P: AsRef<Path>>(wav_path: P, out_path: P) {
    let mut wav = hound::WavReader::open(wav_path).unwrap();
    let mut csv = BufWriter::new(File::create(out_path).unwrap());
    let mut sample_count = 0;
    let mut sample_buf = [0; 8192];
    for s in wav.samples::<i32>() {
        let sample = s.unwrap();
        sample_buf[sample_count] = sample;
        sample_count += 1;
        if sample_count == 8192 {
            sample_count = 0;
            let (_freqs, values) = process_samples(&sample_buf);
            let n = values.len();
            for v in &values[0..(n - 1)] {
                write!(csv, "{v},").unwrap();
            }
            writeln!(csv, "{}", values[n - 1]).unwrap();
            csv.flush().unwrap();
        }
    }
}

pub fn train_model<P: AsRef<Path>>(
    drone_csv: P,
    bg_csv: P,
    out: P,
) -> RandomForestClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>> {
    let mut drone_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(drone_csv)
        .unwrap();
    let mut drone_data = Vec::new();
    for result in drone_reader.deserialize() {
        let record: Vec<f32> = result.unwrap();
        drone_data.push(record);
    }

    let mut bg_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(bg_csv)
        .unwrap();
    let mut bg_data = Vec::new();
    for result in bg_reader.deserialize() {
        let record: Vec<f32> = result.unwrap();
        bg_data.push(record);
    }

    let n_drone = drone_data.len();
    let n_bg = bg_data.len();

    drone_data.append(&mut bg_data);

    let x = DenseMatrix::from_2d_vec(&drone_data).unwrap();
    let mut y = vec![1; n_drone];
    y.append(&mut vec![0; n_bg]);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(42));

    let classifier = RandomForestClassifier::fit(
        &x_train,
        &y_train,
        RandomForestClassifierParameters::default()
            .with_seed(42)
            .with_n_trees(32)
            .with_criterion(SplitCriterion::Gini),
    )
    .unwrap();

    let y_hat = classifier.predict(&x_test).unwrap();

    let acc = ClassificationMetricsOrd::accuracy().get_score(&y_test, &y_hat);
    println!("Accuracy: {acc}");

    let model_bytes = bincode::serialize(&classifier).unwrap();
    File::create(out)
        .and_then(|mut f| f.write_all(&model_bytes))
        .expect("Can not persist the model");

    classifier
}

pub fn load_model<P: AsRef<Path>>(
    model_path: P,
) -> RandomForestClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>> {
    bincode::deserialize_from(BufReader::new(File::open(model_path).unwrap())).unwrap()
}
