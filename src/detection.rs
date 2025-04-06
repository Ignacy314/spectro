use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

use ndarray::Array2;
use smartcore::{
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::basic::matrix::DenseMatrix,
    metrics::{ClassificationMetricsOrd, Metrics},
    model_selection::train_test_split,
};

pub fn train_model<P: AsRef<Path>>(
    drone_csv: P,
    bg_csv: P,
    out: P,
) -> RandomForestClassifier<f32, i32, Array2<f32>, Vec<i32>> {
    let mut drone_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(drone_csv)
        .unwrap();
    let mut drone_data = Vec::new();
    let mut n_drone = 0;
    let mut row_len = 0;
    for result in drone_reader.deserialize() {
        let record: Vec<f32> = result.unwrap();
        row_len = record.len();
        drone_data.extend(record);
        n_drone += 1;
    }

    let mut bg_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(bg_csv)
        .unwrap();
    let mut bg_data = Vec::new();
    let mut n_bg = 0;
    for result in bg_reader.deserialize() {
        let record: Vec<f32> = result.unwrap();
        bg_data.extend(record);
        n_bg += 1;
    }

    drone_data.append(&mut bg_data);

    let x = Array2::from_shape_vec((n_drone + n_bg, row_len), drone_data).unwrap();
    // let x = DenseMatrix::from_2d_vec(&drone_data).unwrap();
    let mut y = vec![1; n_drone];
    y.append(&mut vec![0; n_bg]);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(42));

    let classifier = RandomForestClassifier::fit(
        &x_train,
        &y_train,
        RandomForestClassifierParameters::default()
            .with_seed(42)
            .with_n_trees(32),
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
