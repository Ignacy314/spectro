use clap::{Parser, Subcommand};
use flexi_logger::{Logger, with_thread};
// use spectro::location::Module;

use spectro::{
    DetectionTestArgs, LocationDataArgs, LocationSimArgs, LocationTestArgs, LocationTestI2sArgs,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[allow(clippy::enum_variant_names)]
#[derive(Subcommand)]
enum Commands {
    // Location(LocationArgs),
    LocationTest(LocationTestArgs),
    LocationTestI2s(LocationTestI2sArgs),
    LocationData(LocationDataArgs),
    LocationSim(LocationSimArgs),
    // Detection(DetectionArgs),
    DetectionTest(DetectionTestArgs),
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

fn main() {
    Logger::try_with_env_or_str("info")
        .unwrap()
        .log_to_stderr()
        .format(with_thread)
        .use_utc()
        .start()
        .unwrap();

    let cli = Cli::parse();

    match cli.command {
        // Commands::Detection(_args) => {
        //     // spectro::detection::train_model(args.drone_wav, args.bg_wav, args.out_file);
        // }
        // Commands::Location(_args) => {
        //     // spectro::location::train_model(args.input_dir, args.module, args.out_file);
        // }
        Commands::LocationData(args) => {
            if args.i2s {
                spectro::location_i2s::generate_data_csv(
                    args.input_dir,
                    args.module,
                    args.out_file,
                    args.bad_flights,
                    args.wanted_flights,
                );
            } else {
                spectro::location::generate_data_csv(
                    args.input_dir,
                    args.module,
                    args.out_file,
                    args.bad_flights,
                    args.wanted_flights,
                );
            }
        }
        Commands::LocationTest(args) => {
            spectro::location::test_onnx(
                args.model_file,
                args.input_csv,
                args.plot_path,
                args.module_out,
            );
            // spectro::location::test_avg(
            //     args.model_file,
            //     args.input_dir,
            //     args.module,
            //     args.plot_path,
            // );
        }
        Commands::LocationTestI2s(args) => {
            spectro::location_i2s::test_onnx(
                args.model_file,
                args.plot_path,
                args.module_out,
                args.input_dir,
                args.module,
                args.bad_flights,
                args.wanted_flights,
            );
        }
        Commands::LocationSim(args) => {
            if args.i2s {
                spectro::location_i2s::simulate(args.input_dir, args.modules_csv);
            } else {
                spectro::location::simulate(args.input_dir, args.modules_csv);
            }
        }
        Commands::DetectionTest(args) => {
            spectro::detection::test_onnx(args);
        }
    }
}
