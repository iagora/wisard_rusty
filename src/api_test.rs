extern crate dict_wisard;
extern crate rocket_wisard;
use std::env;
use std::process;

fn main() {
    let config = Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    rocket_wisard::init(
        config
            .number_of_hashtables
            .parse::<u16>()
            .expect("Number of hashtables expects an u16, couldn't convert the input"),
        config
            .address_size
            .parse::<u64>()
            .expect("Address size expects an u64, couldn't convert input"),
        config
            .bleach
            .parse::<u16>()
            .expect("Bleach is an u16, couldn't convert input"),
    );
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
