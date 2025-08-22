use hound::WavReader;
use ndarray::{Array, Array2};
use ort::inputs;
use plotly::{Plot, Scatter, common::Mode};
use regex::Regex;
use serde::Deserialize;
use std::{
    f64,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    thread::sleep,
    time::{Duration, Instant},
};
use tungstenite::connect;

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

fn read_data<P: AsRef<Path>>(
    input_dir: P,
    module: i32,
    bad_flights: Option<Vec<i32>>,
    wanted_flights: Option<Vec<i32>>,
) -> (Array2<f32>, Vec<f64>) {
    println!("reading data for module {module}");

    let module_str = module.to_string();

    let re_wav = Regex::new(r".*\D(\d+)_(\d+)_(\d+)\.wav$").unwrap();
    let re_csv = Regex::new(r".*\D(\d+)\.csv$").unwrap();

    let flights = std::fs::read_dir(input_dir.as_ref().join("i2s")).unwrap();
    let mut flights_wavs: Vec<PathBuf> = flights
        .map(|f| f.unwrap().path().join(&module_str))
        .flat_map(|p| std::fs::read_dir(p).unwrap().map(|d| d.unwrap().path()))
        .filter(|p| {
            let nums = re_wav.captures(p.to_str().unwrap()).unwrap();
            let num = nums[1].parse::<i32>().unwrap() + 1;
            // let mic = nums[2].parse::<i32>().unwrap();
            // let dir = nums[3].parse::<i32>().unwrap();
            let bad = if let Some(bad_flights) = bad_flights.as_ref() {
                !bad_flights.iter().any(|n| *n == num)
            } else {
                true
            };
            let wanted = if let Some(wanted_flights) = wanted_flights.as_ref() {
                wanted_flights.iter().any(|n| *n == num)
            } else {
                true
            };
            bad && wanted
        })
        .collect();
    let mut flights_csvs: Vec<PathBuf> =
        std::fs::read_dir(input_dir.as_ref().join("module_csvs").join(&module_str))
            .unwrap()
            .map(|d| d.unwrap().path())
            .filter(|p| {
                let num: i32 = re_csv.captures(p.to_str().unwrap()).unwrap()[1]
                    .parse()
                    .unwrap();
                let bad = if let Some(bad_flights) = bad_flights.as_ref() {
                    !bad_flights.iter().any(|n| *n == num)
                } else {
                    true
                };
                let wanted = if let Some(wanted_flights) = wanted_flights.as_ref() {
                    wanted_flights.iter().any(|n| *n == num)
                } else {
                    true
                };
                bad && wanted
            })
            .collect();

    assert_eq!(flights_wavs.len(), flights_csvs.len() * 18);

    if flights_wavs.is_empty() {
        log::info!("No flights matching criteria");
        return (Array2::zeros((0, 0)), Vec::new());
    }

    flights_wavs.sort_unstable_by(|a, b| {
        let nums = re_wav.captures(a.to_str().unwrap()).unwrap();
        let a_num = nums[1].parse::<i32>().unwrap() + 1;
        let a_mic = nums[2].parse::<i32>().unwrap();
        let a_dir = nums[3].parse::<i32>().unwrap();
        let nums = re_wav.captures(b.to_str().unwrap()).unwrap();
        let b_num = nums[1].parse::<i32>().unwrap() + 1;
        let b_mic = nums[2].parse::<i32>().unwrap();
        let b_dir = nums[3].parse::<i32>().unwrap();
        a_num
            .cmp(&b_num)
            .then_with(|| a_mic.cmp(&b_mic))
            .then_with(|| a_dir.cmp(&b_dir))
    });

    flights_csvs.sort_unstable_by(|a, b| {
        let a_num: i32 = re_csv.captures(a.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        let b_num: i32 = re_csv.captures(b.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        a_num.cmp(&b_num)
    });

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut row_len = 0;
    for (wav_paths, csv_path) in flights_wavs.chunks(18).zip(flights_csvs.iter()) {
        // eprintln!("{wav_path:?} | {csv_path:?}");
        let mut buffers: [CircularBuffer<8192, i32>; 18] = [const { CircularBuffer::new() }; 18];
        let mut counter = 0;
        let mut wavs: Vec<WavReader<BufReader<File>>> = wav_paths
            .iter()
            .map(|wav_path| hound::WavReader::open(wav_path).unwrap())
            .collect();
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
        // let mut iters = wavs.map(|mut wav| wav.samples::<i32>());

        // let mut samples = Vec::new();
        let mut end = false;

        loop {
            let windows = wavs
                .iter_mut()
                .map(|it| {
                    it.samples::<i32>()
                        .by_ref()
                        .take(2400)
                        .map(|s| s.unwrap())
                        .collect::<Vec<i32>>()
                })
                .collect::<Vec<Vec<i32>>>();
            if windows.iter().any(|w| w.len() < 2400) {
                break;
            }
            for i in 0..2400 {
                windows.iter().zip(buffers.iter_mut()).for_each(|(w, b)| {
                    b.push_back(w[i]);
                });
                counter += 1;
                if buffers[0].is_full() && counter >= 2400 {
                    let Some(distance) = dist_iter.next() else {
                        end = true;
                        break;
                    };
                    counter = 0;
                    let buffer = buffers
                        .iter()
                        .max_by(|a, b| {
                            let a_avg = a.iter().map(|s| *s as f64 / 8192f64).sum::<f64>();
                            let b_avg = b.iter().map(|s| *s as f64 / 8192f64).sum::<f64>();
                            a_avg.total_cmp(&b_avg)
                        })
                        .unwrap();
                    let (_freqs, values) = process_samples(buffer.iter());
                    if row_len != 0 {
                        assert_eq!(row_len, values.len());
                    }
                    row_len = values.len();
                    x_data.extend(values);
                    y_data.push(*distance);
                }
            }
            if end {
                break;
            }
            // let avgs: Vec<f64> = windows
            //     .iter()
            //     .map(|w| w.iter().sum::<i32>() as f64 / 48000.0)
            //     .collect();
            // let mut max_w = windows
            //     .into_iter()
            //     .zip(avgs)
            //     .max_by(|a, b| a.1.total_cmp(&b.1))
            //     .unwrap()
            //     .0;
            // samples.append(&mut max_w);
        }

        // for sample in samples {
        //     // let sample = s.unwrap();
        //     buffer.push_back(sample);
        //     counter += 1;
        //     if buffer.is_full() && counter >= 2400 {
        //         let Some(distance) = dist_iter.next() else {
        //             break;
        //         };
        //         counter = 0;
        //         let (_freqs, values) = process_samples(buffer.iter());
        //         if row_len != 0 {
        //             assert_eq!(row_len, values.len());
        //         }
        //         row_len = values.len();
        //         x_data.extend(values);
        //         y_data.push(*distance);
        //     }
        // }
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
    bad_flights: Option<Vec<i32>>,
    wanted_flights: Option<Vec<i32>>,
) {
    let (x, y) = read_data(input_dir, module, bad_flights, wanted_flights);
    if x.is_empty() || y.is_empty() {
        return;
    }
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

// fn read_data_csv<P: AsRef<Path>>(csv_path: P) -> (Vec<Vec<f32>>, Vec<f64>) {
//     let mut csv = csv::ReaderBuilder::new()
//         .has_headers(false)
//         .from_path(csv_path)
//         .unwrap();
//
//     let mut x = Vec::new();
//     let mut y = Vec::new();
//
//     for result in csv.deserialize() {
//         let (y_data, x_data): (f64, Vec<f32>) = result.unwrap();
//         y.push(y_data);
//         x.push(x_data);
//     }
//
//     (x, y)
// }

pub fn test_onnx<P: AsRef<Path>>(
    model_path: P,
    plot_path: P,
    module_out: Option<String>,
    input_dir: P,
    module: i32,
    bad_flights: Option<Vec<i32>>,
    wanted_flights: Option<Vec<i32>>,
) {
    const ANGLES: [f64; 9] = [30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0];

    println!("loading onnx model");
    let model = load_onnx(model_path);

    println!("reading data for module {module}");

    let module_str = module.to_string();

    let re_wav = Regex::new(r".*\D(\d+)_(\d+)_(\d+)\.wav$").unwrap();
    let re_csv = Regex::new(r".*\D(\d+)\.csv$").unwrap();

    let flights = std::fs::read_dir(input_dir.as_ref().join("i2s")).unwrap();
    let mut flights_wavs: Vec<PathBuf> = flights
        .map(|f| f.unwrap().path().join(&module_str))
        .flat_map(|p| std::fs::read_dir(p).unwrap().map(|d| d.unwrap().path()))
        .filter(|p| {
            let nums = re_wav.captures(p.to_str().unwrap()).unwrap();
            let num = nums[1].parse::<i32>().unwrap() + 1;
            // let mic = nums[2].parse::<i32>().unwrap();
            // let dir = nums[3].parse::<i32>().unwrap();
            let bad = if let Some(bad_flights) = bad_flights.as_ref() {
                !bad_flights.iter().any(|n| *n == num)
            } else {
                true
            };
            let wanted = if let Some(wanted_flights) = wanted_flights.as_ref() {
                wanted_flights.iter().any(|n| *n == num)
            } else {
                true
            };
            bad && wanted
        })
        .collect();
    let mut flights_csvs: Vec<PathBuf> =
        std::fs::read_dir(input_dir.as_ref().join("module_csvs").join(&module_str))
            .unwrap()
            .map(|d| d.unwrap().path())
            .filter(|p| {
                let num: i32 = re_csv.captures(p.to_str().unwrap()).unwrap()[1]
                    .parse()
                    .unwrap();
                let bad = if let Some(bad_flights) = bad_flights.as_ref() {
                    !bad_flights.iter().any(|n| *n == num)
                } else {
                    true
                };
                let wanted = if let Some(wanted_flights) = wanted_flights.as_ref() {
                    wanted_flights.iter().any(|n| *n == num)
                } else {
                    true
                };
                bad && wanted
            })
            .collect();

    assert_eq!(flights_wavs.len(), flights_csvs.len() * 18);

    if flights_wavs.is_empty() {
        log::info!("No flights matching criteria");
        return;
    }

    flights_wavs.sort_unstable_by(|a, b| {
        let nums = re_wav.captures(a.to_str().unwrap()).unwrap();
        let a_num = nums[1].parse::<i32>().unwrap() + 1;
        let a_mic = nums[2].parse::<i32>().unwrap();
        let a_dir = nums[3].parse::<i32>().unwrap();
        let nums = re_wav.captures(b.to_str().unwrap()).unwrap();
        let b_num = nums[1].parse::<i32>().unwrap() + 1;
        let b_mic = nums[2].parse::<i32>().unwrap();
        let b_dir = nums[3].parse::<i32>().unwrap();
        a_num
            .cmp(&b_num)
            .then_with(|| a_mic.cmp(&b_mic))
            .then_with(|| a_dir.cmp(&b_dir))
    });

    flights_csvs.sort_unstable_by(|a, b| {
        let a_num: i32 = re_csv.captures(a.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        let b_num: i32 = re_csv.captures(b.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        a_num.cmp(&b_num)
    });

    let mut dists_h = Vec::new();
    let mut angles_h = Vec::new();
    let mut dists_v = Vec::new();
    let mut angles_v = Vec::new();

    let mut y = Vec::new();
    let mut row_len = 0;
    for (wav_paths, csv_path) in flights_wavs.chunks(18).zip(flights_csvs.iter()) {
        // eprintln!("{wav_path:?} | {csv_path:?}");
        let mut buffers: [CircularBuffer<8192, i32>; 18] = [const { CircularBuffer::new() }; 18];
        let mut counter = 0;
        let mut wavs: Vec<WavReader<BufReader<File>>> = wav_paths
            .iter()
            .map(|wav_path| hound::WavReader::open(wav_path).unwrap())
            .collect();
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
        // let mut iters = wavs.map(|mut wav| wav.samples::<i32>());

        // let mut samples = Vec::new();
        let mut end = false;

        loop {
            let windows = wavs
                .iter_mut()
                .map(|it| {
                    it.samples::<i32>()
                        .by_ref()
                        .take(2400)
                        .map(|s| s.unwrap())
                        .collect::<Vec<i32>>()
                })
                .collect::<Vec<Vec<i32>>>();
            if windows.iter().any(|w| w.len() < 2400) {
                break;
            }
            for i in 0..2400 {
                windows.iter().zip(buffers.iter_mut()).for_each(|(w, b)| {
                    b.push_back(w[i]);
                });
                counter += 1;
                if buffers[0].is_full() && counter >= 2400 {
                    let Some(distance) = dist_iter.next() else {
                        end = true;
                        break;
                    };
                    counter = 0;
                    let mut dist_h = f64::MAX;
                    let mut angle_h = -1f64;
                    for (i, buffer) in buffers[0..9].iter().enumerate() {
                        let (_freqs, values) = process_samples(buffer.iter());
                        if row_len != 0 {
                            assert_eq!(row_len, values.len());
                        }
                        row_len = values.len();
                        let x = Array::from_shape_vec((1, values.len()), values).unwrap();
                        let outputs = model.run(inputs![x].unwrap()).unwrap();
                        let y_pred: f64 = *outputs["variable"]
                            .try_extract_tensor()
                            .unwrap()
                            .first()
                            .unwrap();
                        if y_pred < dist_h {
                            dist_h = y_pred;
                            angle_h = ANGLES[i];
                        }
                    }
                    let mut dist_v = f64::MAX;
                    let mut angle_v = -1f64;
                    for (i, buffer) in buffers[9..18].iter().enumerate() {
                        let (_freqs, values) = process_samples(buffer.iter());
                        if row_len != 0 {
                            assert_eq!(row_len, values.len());
                        }
                        row_len = values.len();
                        let x = Array::from_shape_vec((1, values.len()), values).unwrap();
                        let outputs = model.run(inputs![x].unwrap()).unwrap();
                        let y_pred: f64 = *outputs["variable"]
                            .try_extract_tensor()
                            .unwrap()
                            .first()
                            .unwrap();
                        if y_pred < dist_h {
                            dist_v = y_pred;
                            angle_v = ANGLES[i];
                        }
                    }

                    dists_h.push(dist_h);
                    angles_h.push(angle_h);
                    dists_v.push(dist_v);
                    angles_v.push(angle_v);

                    y.push(*distance);
                }
            }
            if end {
                break;
            }
        }
    }

    println!("testing onnx model");

    println!("number of outputs: {}", dists_h.len());

    if let Some(module_out) = module_out {
        // let mac = format!("sim.{}", module.n);
        // let ip = mac.clone();
        std::fs::create_dir_all(Path::new(&module_out).parent().unwrap()).unwrap();
        let mut csv = BufWriter::new(File::create(module_out).unwrap());
        writeln!(csv, "dist").unwrap();
        for dist in dists_h.iter() {
            writeln!(csv, "{dist}").unwrap();
        }
        // writeln!(csv, "mac,ip,lat,lon,drone,dist").unwrap();
        // for dist in y_avg.iter() {
        //     writeln!(csv, "{},{},{},{},true,{dist}", &mac, &ip, module.lat, module.lon).unwrap();
        // }
    }

    let dists_h_avg: Vec<f64> = dists_h
        .windows(20)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect();
    let dists_v_avg: Vec<f64> = dists_v
        .windows(20)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect();

    let x: Vec<usize> = (0..dists_h_avg.len()).collect();
    let mut plot = Plot::new();
    let y_test_plot = Scatter::new(x.clone(), y);
    let y_hat_h_plot = Scatter::new(x.clone(), dists_h_avg).mode(Mode::Markers);
    let y_hat_v_plot = Scatter::new(x, dists_v_avg).mode(Mode::Markers);
    plot.add_traces(vec![y_hat_h_plot, y_hat_v_plot, y_test_plot]);
    plot.write_html(plot_path);

    // let x_tract =
    //     tract_ndarray::Array2::from_shape_vec((x_shape[0], x_shape[1]), x.into_raw_vec()).unwrap();
    // let x_tensor: Tensor = x_tract.into();
    //
    // let result = model.run(tvec!(x_tensor.into())).unwrap();
    //
    // println!("{:?}", result[0]);
}

#[derive(Deserialize)]
struct ModuleRecord {
    module: i32,
    lat: f64,
    lon: f64,
}

struct Module {
    _module: i32,
    mac: String,
    ip: String,
    lat: f64,
    lon: f64,
}

pub fn simulate<P: AsRef<Path>>(input_dir: P, modules_csv: P) {
    let re_csv = Regex::new(r".*\D(\d+)\.csv$").unwrap();

    let mut csvs: Vec<PathBuf> = std::fs::read_dir(input_dir)
        .unwrap()
        .map(|d| d.unwrap().path())
        .collect();
    csvs.sort_unstable_by(|a, b| {
        let a_num: i32 = re_csv.captures(a.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        let b_num: i32 = re_csv.captures(b.to_str().unwrap()).unwrap()[1]
            .parse()
            .unwrap();
        a_num.cmp(&b_num)
    });

    let mut modules_csv = csv::Reader::from_path(modules_csv).unwrap();

    let mut modules = Vec::new();
    for module in modules_csv.deserialize() {
        let r: ModuleRecord = module.unwrap();
        let mac = format!("sim.{}", r.module);
        modules.push(Module {
            _module: r.module,
            mac: mac.clone(),
            ip: mac,
            lat: r.lat,
            lon: r.lon,
        });
    }

    assert_eq!(modules.len(), csvs.len());

    let mut readers = Vec::new();
    let mut desers = Vec::new();
    for csv in csvs {
        let reader = csv::Reader::from_path(csv).unwrap();
        readers.push(reader);
    }
    for reader in readers.iter_mut() {
        desers.push(reader.deserialize::<f64>());
    }

    let read_period = Duration::from_millis(50);

    let (mut socket, _response) = match connect("ws://10.66.66.1:3012/socket") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Drone WebSocket connection error: {e}");
            return;
        }
    };
    println!("Drone WebSocket connected");

    loop {
        let start = Instant::now();

        let distances: Vec<f64> = desers
            .iter_mut()
            .map(|d| d.next().unwrap().unwrap())
            .collect();

        for (module, dist) in modules.iter().zip(distances.iter()) {
            let msg =
                format!("{}|{}|{}|{}|true|{dist}", &module.mac, &module.ip, module.lat, module.lon);
            log::info!("{msg}");
            socket.send(tungstenite::Message::Text(msg.into())).unwrap();
        }

        sleep(read_period.saturating_sub(start.elapsed()));
    }
}
