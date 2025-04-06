use clap::Parser;
#[allow(unused_imports)]
use spectro::{detection::train_model, location, wav_to_csv};

#[derive(clap::Parser)]
struct Args {
    #[arg(short, long)]
    drone_wav: String,
    #[arg(short, long)]
    drone_csv: String,
    #[arg(short, long)]
    bg_wav: Option<String>,
    #[arg(short, long)]
    bg_csv: Option<String>,
    #[arg(short, long)]
    model: Option<String>,
    #[arg(short, long)]
    out: String,
}

fn main() {
    let args = Args::parse();

    // wav_to_csv(args.drone_wav, args.drone_csv.clone());
    // wav_to_csv(args.bg_wav, args.bg_csv.clone());

    // train_model(args.drone_csv, args.bg_csv, args.out);
    // location::train_model(args.drone_wav, args.drone_csv, args.out);
    location::test_avg(args.model.as_ref().unwrap(), &args.drone_wav, &args.drone_csv, &args.out);

    // wav_to_csv(
    //     "/home/test/mnt/dane/29-03-25_2/combined/D4_29032025.wav",
    //     "/home/test/mnt/dane/29-03-25_2/rust_csvs/D4_29032025.csv",
    // );
    // wav_to_csv(
    //     "/home/test/mnt/dane/29-03-25_2/combined/B4_29032025.wav",
    //     "/home/test/mnt/dane/29-03-25_2/rust_csvs/B4_29032025.csv",
    // );
    //
    // let _model = train_model(
    //     "/home/test/mnt/dane/29-03-25_2/rust_csvs/D4_29032025.csv",
    //     "/home/test/mnt/dane/29-03-25_2/rust_csvs/B4_29032025.csv",
    //     "/home/test/mnt/dane/29-03-25_2/rust_csvs/DB4_29032025.model",
    // );

    // wav_to_csv(
    //     "/home/iluvatar/andros/models/D4_29032025.wav",
    //     "/home/iluvatar/andros/models/D4_29032025.csv",
    // );
    // wav_to_csv(
    //     "/home/iluvatar/andros/models/B4_29032025.wav",
    //     "/home/iluvatar/andros/models/B4_29032025.csv",
    // );

    // let _model = train_model(
    //     "/home/iluvatar/andros/models/old/D4_29032025.csv",
    //     "/home/iluvatar/andros/models/old/B4_29032025.csv",
    //     "/home/iluvatar/andros/models/old/DB4_29032025.model",
    // );
    // let _model = train_model(
    //     "/home/iluvatar/andros/models/D4_29032025.csv",
    //     "/home/iluvatar/andros/models/B4_29032025.csv",
    //     "/home/iluvatar/andros/models/DB4_29032025.model",
    // );
}
