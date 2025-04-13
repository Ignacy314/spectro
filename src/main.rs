use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Location(LocationArgs),
    LocationTest(LocationTestArgs),
    LocationData(LocationDataArgs),
    // Detection(DetectionArgs),
}

// #[derive(clap::Args)]
// struct LocationArgs {
//     #[arg(short, long)]
//     input_dir: String,
//     #[arg(short, long)]
//     module: i32,
//     #[arg(short, long)]
//     out_file: String,
// }

#[derive(clap::Args)]
struct LocationDataArgs {
    #[arg(short, long)]
    input_dir: String,
    #[arg(short, long)]
    module: i32,
    #[arg(short, long)]
    out_file: String,
    #[arg(short, long)]
    bad_flights: Vec<i32>,
}

#[derive(clap::Args)]
struct LocationTestArgs {
    #[arg(short, long)]
    model_file: String,
    #[arg(short, long)]
    input_csv: String,
    #[arg(short, long)]
    plot_path: String,
}

// #[derive(clap::Args)]
// struct DetectionArgs {
//     #[arg(short, long)]
//     drone_wav: String,
//     #[arg(short, long)]
//     bg_wav: String,
//     #[arg(short, long)]
//     out_file: String,
// }

fn main() {
    let cli = Cli::parse();

    match cli.command {
        // Commands::Detection(_args) => {
        //     // spectro::detection::train_model(args.drone_wav, args.bg_wav, args.out_file);
        // }
        // Commands::Location(_args) => {
        //     // spectro::location::train_model(args.input_dir, args.module, args.out_file);
        // }
        Commands::LocationData(args) => {
            spectro::location::generate_data_csv(
                args.input_dir,
                args.module,
                args.out_file,
                args.bad_flights,
            );
        }
        Commands::LocationTest(args) => {
            spectro::location::test_onnx(args.model_file, args.input_csv, args.plot_path);
            // spectro::location::test_avg(
            //     args.model_file,
            //     args.input_dir,
            //     args.module,
            //     args.plot_path,
            // );
        }
    }
}
