extern crate actix_wisard;
extern crate wisard;

use std::process;
use wisard::wisard_traits::WisardNetwork;

fn main() {
    let wis = Box::new(wisard::dict_wisard::Wisard::<u8>::new())
        as Box<dyn WisardNetwork<u8> + Send + Sync + 'static>;
    if let Err(e) = actix_wisard::run(wis) {
        eprintln!("Application error: {}", e);

        process::exit(1);
    }
}
