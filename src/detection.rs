// use std::{
//     fs::File,
//     io::{BufReader, Write},
//     path::Path,
// };
//
// use circular_buffer::CircularBuffer;
// use ndarray::Array2;
// use smartcore::{
//     ensemble::random_forest_classifier::{
//         RandomForestClassifier, RandomForestClassifierParameters,
//     },
//     linalg::basic::matrix::DenseMatrix,
//     metrics::{ClassificationMetricsOrd, Metrics},
//     model_selection::train_test_split,
// };
//
// use crate::process_samples;

// pub fn train_model<P: AsRef<Path>>(
//     drone_wav_path: P,
//     bg_wav_path: P,
//     out: P,
// ) -> RandomForestClassifier<f32, i32, Array2<f32>, Vec<i32>> {
//     let mut x_data = Vec::new();
//     let mut row_len = 0;
//
//     {
//         let mut wav = hound::WavReader::open(drone_wav_path).unwrap();
//         let mut buffer: CircularBuffer<8192, i32> = CircularBuffer::new();
//         let mut counter = 0;
//         for s in wav.samples::<i32>() {
//             let sample = s.unwrap();
//             buffer.push_back(sample);
//             counter += 1;
//             if buffer.is_full() && counter >= 2400 {
//                 counter = 0;
//                 let (_freqs, values) = process_samples(buffer.iter());
//                 if row_len != 0 {
//                     assert_eq!(row_len, values.len());
//                 }
//                 row_len = values.len();
//                 x_data.extend(values);
//             }
//         }
//     }
//     let n_drone = x_data.len() / row_len;
//
//     {
//         let mut wav = hound::WavReader::open(bg_wav_path).unwrap();
//         let mut buffer: CircularBuffer<8192, i32> = CircularBuffer::new();
//         let mut counter = 0;
//         for s in wav.samples::<i32>() {
//             let sample = s.unwrap();
//             buffer.push_back(sample);
//             counter += 1;
//             if buffer.is_full() && counter >= 2400 {
//                 counter = 0;
//                 let (_freqs, values) = process_samples(buffer.iter());
//                 if row_len != 0 {
//                     assert_eq!(row_len, values.len());
//                 }
//                 row_len = values.len();
//                 x_data.extend(values);
//             }
//         }
//     }
//
//     let n = x_data.len() / row_len;
//     let n_bg = n - n_drone;
//
//     let mut y = vec![1; n_drone];
//     y.extend(vec![0; n_bg]);
//
//     // let n_y = y.len();
//     // match n_y.cmp(&n) {
//     //     std::cmp::Ordering::Less => {
//     //         x_data.truncate(n_y * row_len);
//     //         n = n_y;
//     //     }
//     //     std::cmp::Ordering::Greater => {
//     //         y.truncate(n);
//     //     }
//     //     _ => {}
//     // }
//
//     // let mut drone_reader = csv::ReaderBuilder::new()
//     //     .has_headers(false)
//     //     .from_path(drone_csv)
//     //     .unwrap();
//     // let mut drone_data = Vec::new();
//     // let mut n_drone = 0;
//     // let mut row_len = 0;
//     // for result in drone_reader.deserialize() {
//     //     let record: Vec<f32> = result.unwrap();
//     //     row_len = record.len();
//     //     drone_data.extend(record);
//     //     n_drone += 1;
//     // }
//     //
//     // let mut bg_reader = csv::ReaderBuilder::new()
//     //     .has_headers(false)
//     //     .from_path(bg_csv)
//     //     .unwrap();
//     // let mut bg_data = Vec::new();
//     // let mut n_bg = 0;
//     // for result in bg_reader.deserialize() {
//     //     let record: Vec<f32> = result.unwrap();
//     //     bg_data.extend(record);
//     //     n_bg += 1;
//     // }
//     //
//     // drone_data.append(&mut bg_data);
//
//     let x = Array2::from_shape_vec((n, row_len), x_data).unwrap();
//     // let x = DenseMatrix::from_2d_vec(values)
//     // let x = DenseMatrix::from_2d_vec(&drone_data).unwrap();
//     // let mut y = vec![1; n_drone];
//     // y.append(&mut vec![0; n_bg]);
//     let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.8, true, Some(42));
//
//     let classifier = RandomForestClassifier::fit(
//         &x_train,
//         &y_train,
//         RandomForestClassifierParameters::default()
//             .with_seed(42)
//             .with_n_trees(16),
//     )
//     .unwrap();
//
//     let y_hat = classifier.predict(&x_test).unwrap();
//
//     let acc = ClassificationMetricsOrd::accuracy().get_score(&y_test, &y_hat);
//     println!("Accuracy: {acc}");
//
//     let model_bytes = bincode::serialize(&classifier).unwrap();
//     File::create(out)
//         .and_then(|mut f| f.write_all(&model_bytes))
//         .expect("Can not persist the model");
//
//     classifier
// }

// pub fn load_model<P: AsRef<Path>>(
//     model_path: P,
// ) -> RandomForestClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>> {
//     bincode::deserialize_from(BufReader::new(File::open(model_path).unwrap())).unwrap()
// }

use circular_buffer::CircularBuffer;
use ndarray::{Array2, ArrayViewD};
use ort::inputs;

use crate::{DetectionTestArgs, load_onnx, process_samples};

pub fn test_onnx(args: DetectionTestArgs) {
    let detection_model = load_onnx(args.model);
    let mut detections: CircularBuffer<20, u8> = CircularBuffer::from([0; 20]);
    let mut count = 0;
    let mut detection_count = 0;
    let mut single_detection_count = 0;

    let mut wav = hound::WavReader::open(args.input_wav).unwrap();
    let mut sample_count = 0;
    let mut sample_buf = [0; 8192];
    for s in wav.samples::<i32>() {
        let sample = s.unwrap();
        sample_buf[sample_count] = sample;
        sample_count += 1;
        if sample_count == 8192 {
            sample_count = 0;
            let (_freqs, values) = process_samples(sample_buf.iter());
            if let Ok(x) = Array2::from_shape_vec((1, values.len()), values)
                && let Ok(outputs) = detection_model.run(inputs![x.clone()].unwrap())
            {
                // println!("{outputs:?}");
                let pred: ArrayViewD<i64> = outputs["output_label"].try_extract_tensor().unwrap();
                let pred_len = pred.len();
                let pred = pred.into_shape_with_order(pred_len).unwrap()[0];
                // detections.push_back(if pred[0] == 1 { 1 } else { 0 });
                detections.push_back(pred as u8);
                let drone_predicted = detections.iter().sum::<u8>() > 1;
                count += 1;
                if drone_predicted {
                    detection_count += 1;
                }
                single_detection_count += pred;
                // println!("Drone detected: {pred} {drone_predicted}");
            }
        }
    }
    println!(
        "Total count =  {count} | Drone detected count = {detection_count} | Single count = {single_detection_count}"
    );
}
