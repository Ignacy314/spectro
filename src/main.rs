use spectro::{train_model, wav_to_csv};

fn main() {
    wav_to_csv(
        "/home/test/mnt/dane/29-03-25_2/combined/D4_29032025.wav",
        "/home/test/mnt/dane/29-03-25_2/rust_csvs/D4_29032025.csv",
    );
    wav - to_csv(
        "/home/test/mnt/dane/29-03-25_2/combined/B4_29032025.wav",
        "/home/test/mnt/dane/29-03-25_2/rust_csvs/B4_29032025.csv",
    );

    let _model = train_model(
        "/home/test/mnt/dane/29-03-25_2/rust_csvs/D4_29032025.csv",
        "/home/test/mnt/dane/29-03-25_2/rust_csvs/D4_29032025.csv",
        "/home/test/mnt/dane/29-03-25_2/rust_csvs/DB4_29032025.model",
    );
}
