#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate rocket;
mod wisard;

#[get("/")]
fn index() -> &'static str {
    "abracadabra..bitches!"
}

fn main() {
    rocket::ignite().mount("/", routes![index]).launch();
}
