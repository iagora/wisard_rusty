#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use]
extern crate rocket;
extern crate dict_wisard;

use rocket::response::Stream;
use rocket::State;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

pub fn ignite() {
    let wis = Arc::new(Mutex::new(dict_wisard::wisard::Wisard::<u8>::new()));
    rocket::ignite()
        .mount("/", routes![train, classify, save, load, erase, new])
        .manage(wis)
        .launch();
}

#[post("/new?<hashtables>&<addresses>&<bleach>")]
pub fn new(
    wis: State<Arc<Mutex<dict_wisard::wisard::Wisard<u8>>>>,
    hashtables: u16,
    addresses: u16,
    bleach: u16,
) {
    wis.lock()
        .unwrap()
        .erase_and_change_hyperparameters(hashtables, addresses, bleach);
}

#[post(
    "/with_model?<hashtables>&<addresses>&<bleach>",
    format = "multipart",
    data = "<weights>"
)]
pub fn with_model(
    wis: State<Arc<Mutex<dict_wisard::wisard::Wisard<u8>>>>,
    hashtables: u16,
    addresses: u16,
    bleach: u16,
    weights: FileMultipart,
) {
    let mut unlocked_wis = wis.lock().unwrap();
    unlocked_wis.erase_and_change_hyperparameters(hashtables, addresses, bleach);
    unlocked_wis.load(&weights.file);
}

#[post("/train?<label>", format = "multipart", data = "<image>")]
pub fn train(
    wis: State<Arc<Mutex<dict_wisard::wisard::Wisard<u8>>>>,
    label: String,
    image: FileMultipart,
) {
    let mut unlocked_wis = wis.lock().unwrap();
    unlocked_wis.train(image.file, label);
}

#[post("/classify", format = "multipart", data = "<image>")]
pub fn classify(
    wis: State<Arc<Mutex<dict_wisard::wisard::Wisard<u8>>>>,
    image: FileMultipart,
) -> String {
    let (label, _, _) = wis.lock().unwrap().classify(image.file);
    return label;
}

#[get("/model")]
pub fn save(wis: State<Arc<Mutex<dict_wisard::wisard::Wisard<u8>>>>) -> Stream<Cursor<Vec<u8>>> {
    let encoded: Vec<u8> = wis.lock().unwrap().save();
    Stream::from(Cursor::new(encoded))
}
#[post("/model", format = "multipart", data = "<weights>")]
pub fn load(wis: State<Arc<Mutex<dict_wisard::wisard::Wisard<u8>>>>, weights: FileMultipart) {
    wis.lock().unwrap().load(&weights.file);
}
#[delete("/model")]
pub fn erase(wis: State<Arc<Mutex<dict_wisard::wisard::Wisard<u8>>>>) {
    wis.lock().unwrap().erase();
}

use multipart::server::Multipart; // 0.16.1, default-features = false, features = ["server"]
use rocket::{
    data::{Data, FromData, Outcome, Transform, Transformed},
    post, routes, Request,
}; // 0.4.2
use std::io::Read;

pub struct FileMultipart {
    file: Vec<u8>,
}

impl<'a> FromData<'a> for FileMultipart {
    type Owned = Vec<u8>;
    type Borrowed = [u8];
    type Error = ();

    fn transform(_request: &Request, data: Data) -> Transform<Outcome<Self::Owned, Self::Error>> {
        let mut d = Vec::new();
        data.stream_to(&mut d).expect("Unable to read");

        Transform::Owned(Outcome::Success(d))
    }

    fn from_data(request: &Request, outcome: Transformed<'a, Self>) -> Outcome<Self, Self::Error> {
        let d = outcome.owned()?;

        let ct = request
            .headers()
            .get_one("Content-Type")
            .expect("no content-type");
        let idx = ct.find("boundary=").expect("no boundary");
        let boundary = &ct[(idx + "boundary=".len())..];

        let mut mp = Multipart::with_body(&d[..], boundary);

        // Custom implementation parts
        let mut file = None;

        mp.foreach_entry(|mut entry| match &*entry.headers.name {
            "file" => {
                let mut d = Vec::new();
                entry.data.read_to_end(&mut d).expect("not file");
                file = Some(d);
            }
            other => panic!("No known key {}", other),
        })
        .expect("Unable to iterate");

        let v = FileMultipart {
            file: file.expect("file not set"),
        };

        // End custom

        Outcome::Success(v)
    }
}
