mod mnist;
mod wisard;

use std::env;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    // println!("Dataset: {}", config.dataset_name);
    println!("Address size: {}", config.address_size);

    let mut wis = wisard::Wisard::new(28, config.address_size.parse::<u64>()?);

    let now = Instant::now();

    let filename = format!("data/{}-labels-idx1-ubyte.gz", "train");
    let label_data = &mnist::MnistData::new(&(File::open(filename))?)?;
    let filename = format!("data/{}-images-idx3-ubyte.gz", "train");
    let images_data = &mnist::MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Vec<u8>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        images.push(image_data);
    }

    println!(
        "Parsing the dataset took: {} milliseconds",
        now.elapsed().as_millis()
    );

    let classifications: Vec<u8> = label_data.data.clone();

    let now = Instant::now();
    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        wis.train(image, classification.to_string());
    }
    println!("Training took: {} seconds", now.elapsed().as_secs());

    let filename = format!("data/{}-labels-idx1-ubyte.gz", "t10k");
    let label_data = &mnist::MnistData::new(&(File::open(filename))?)?;
    let filename = format!("data/{}-images-idx3-ubyte.gz", "t10k");
    let images_data = &mnist::MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Vec<u8>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        images.push(image_data);
    }

    let classifications: Vec<u8> = label_data.data.clone();

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
    println!("Classifying took: {} seconds", now.elapsed().as_secs());

    println!("Accuracy: {}", hit as f64 / count as f64);

    Ok(())
}

pub struct Config {
    // pub filename: String,
    pub address_size: String,
}

impl Config {
    pub fn new(mut args: env::Args) -> Result<Config, &'static str> {
        args.next();
        // let filename = match args.next() {
        //     Some(arg) => {
        //         if arg == "mnist" {
        //             arg
        //         } else {
        //             return Err("We only run for mnist at the moment :/");
        //         }
        //     }
        //     None => return Err("We only run for mnist at the moment :/"),
        // };

        let address_size = match args.next() {
            Some(arg) => arg,
            None => return Err("Didn't get address size"),
        };

        Ok(Config {
            // filename,
            address_size,
        })
    }
}
