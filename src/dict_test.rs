use dict_wisard::{mnist, wisard};
use std::env;
use std::error::Error;
use std::fs::File;
use std::process;
use std::time::Instant;

fn main() {
    let config = Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    if let Err(e) = run(config) {
        eprintln!("Application error: {}", e);

        process::exit(1);
    }
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    println!("Rusty WiSARD - MNIST 🦀🦀🦀");
    println!("Number of hashtables: {}", config.number_of_hashtables);
    println!("Address size: {}", config.address_size);
    println!("Bleaching: {}", config.bleach);

    let mut wis = wisard::Wisard::with_params(
        config.number_of_hashtables.parse::<u16>()?,
        config.address_size.parse::<u16>()?,
        config.bleach.parse::<u16>()?,
    );
    println!("\n-----------------\nTraining\n-----------------");
    let now = Instant::now();

    let label_data =
        &mnist::MnistData::new(&(File::open("data/mnist/train-labels-idx1-ubyte.gz"))?)?;
    let images_data =
        &mnist::MnistData::new(&(File::open("data/mnist/train-images-idx3-ubyte.gz"))?)?;
    let mut images: Vec<Vec<u8>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        images.push(image_data);
    }

    let classifications: Vec<u8> = label_data.data.clone();

    println!("Training data has {} images", classifications.len());
    println!(
        "Parsing the training dataset took: {} milliseconds",
        now.elapsed().as_millis()
    );

    let now = Instant::now();
    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        wis.train(image, classification.to_string());
    }
    println!("Training took: {} milliseconds", now.elapsed().as_millis());

    println!("\n-----------------\nTesting\n-----------------");
    let now = Instant::now();
    let label_data =
        &mnist::MnistData::new(&(File::open("data/mnist/t10k-labels-idx1-ubyte.gz"))?)?;
    let images_data =
        &mnist::MnistData::new(&(File::open("data/mnist/t10k-images-idx3-ubyte.gz"))?)?;
    let mut images: Vec<Vec<u8>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        images.push(image_data);
    }
    let classifications: Vec<u8> = label_data.data.clone();

    println!("Testing data has {} images", classifications.len());
    println!(
        "Parsing the test dataset took: {} milliseconds",
        now.elapsed().as_millis()
    );
    let mut hit: u64 = 0;
    let mut count: u64 = 0;

    let now = Instant::now();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        let tuple: (String, f64, f64);
        tuple = wis.classify(image);
        if tuple.0 == classification.to_string() {
            hit += 1;
        }
        count += 1;
    }
    println!("Testing took: {} milliseconds", now.elapsed().as_millis());

    println!("Accuracy: {}", hit as f64 / count as f64);

    Ok(())
}

pub struct Config {
    // pub filename: String,
    pub number_of_hashtables: String,
    pub address_size: String,
    pub bleach: String,
}

impl Config {
    pub fn new(mut args: env::Args) -> Result<Config, &'static str> {
        args.next();

        let number_of_hashtables = match args.next() {
            Some(arg) => arg,
            None => return Err("Didn't get number of hashtables size"),
        };

        let address_size = match args.next() {
            Some(arg) => arg,
            None => return Err("Didn't get address size"),
        };

        let bleach = match args.next() {
            Some(arg) => arg,
            None => String::from("0"),
        };

        Ok(Config {
            number_of_hashtables,
            address_size,
            bleach,
        })
    }
}
