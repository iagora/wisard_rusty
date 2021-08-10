extern crate actix_wisard;
extern crate wisard;

use std::process;

fn main() {
    if let Err(e) = actix_wisard::run(wisard::dict_wisard::Wisard::<u8>::new()) {
        eprintln!("Application error: {}", e);

        process::exit(1);
    }
}
