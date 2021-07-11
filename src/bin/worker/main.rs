#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate rocket;
// use rocket::response::content::Json;
mod worker;

#[get("/")]
fn index() -> &'static str {
    "abracadabra..bitches!"
}

fn main() {
    let d = worker::Discriminator::new(28);
    rocket::ignite().mount("/", routes![index]).launch();
}
