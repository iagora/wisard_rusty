#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use]
extern crate rocket;
extern crate wisard;

use wisard::wisard_traits::WisardNetwork;

use rocket::response::Stream;
use rocket::State;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

pub fn ignite() {
    let wis = Arc::new(Mutex::new(wisard::dict_wisard::Wisard::<u8>::new()));
    rocket::ignite()
        .mount(
            "/",
            routes![new, with_model, train, classify, save, load, erase],
        )
        .manage(wis)
        .launch();
}

#[post("/new?<hashtables>&<addresses>&<bleach>")]
pub fn new(
    wis: State<Arc<Mutex<wisard::dict_wisard::Wisard<u8>>>>,
    hashtables: u16,
    addresses: u16,
    bleach: u16,
) {
    wis.lock()
        .unwrap()
        .erase_and_change_hyperparameters(hashtables, addresses, bleach);
}

#[post("/with_model", format = "multipart", data = "<model>")]
pub fn with_model(wis: State<Arc<Mutex<wisard::dict_wisard::Wisard<u8>>>>, model: ModelMultipart) {
    let mut unlocked_wis = wis.lock().unwrap();
    unlocked_wis.erase_and_change_hyperparameters(
        model.number_of_hashtables,
        model.addr_length,
        model.bleach,
    );
    unlocked_wis.load(&model.weights).unwrap();
}

#[post("/train", format = "multipart", data = "<image>")]
pub fn train(wis: State<Arc<Mutex<wisard::dict_wisard::Wisard<u8>>>>, image: TrainImageMultipart) {
    let mut unlocked_wis = wis.lock().unwrap();
    unlocked_wis.train(image.image, image.label).unwrap();
}

#[post("/classify", format = "multipart", data = "<image>")]
pub fn classify(
    wis: State<Arc<Mutex<wisard::dict_wisard::Wisard<u8>>>>,
    image: ClassifyImageMultipart,
) -> String {
    return wis.lock().unwrap().classify(image.image).unwrap();
}
#[get("/model")]
pub fn save(wis: State<Arc<Mutex<wisard::dict_wisard::Wisard<u8>>>>) -> Stream<Cursor<Vec<u8>>> {
    let encoded: Vec<u8> = wis.lock().unwrap().save().unwrap();
    Stream::from(Cursor::new(encoded))
}
#[post("/model", format = "multipart", data = "<weights>")]
pub fn load(wis: State<Arc<Mutex<wisard::dict_wisard::Wisard<u8>>>>, weights: ModelMultipart) {
    wis.lock().unwrap().load(&weights.weights).unwrap();
}
#[delete("/model")]
pub fn erase(wis: State<Arc<Mutex<wisard::dict_wisard::Wisard<u8>>>>) {
    wis.lock().unwrap().erase();
}

use multipart::server::Multipart;
use rocket::{
    data::{Data, FromData, Outcome, Transform, Transformed},
    post, routes, Request,
};
use std::io::Read;

pub struct ModelMultipart {
    number_of_hashtables: u16,
    addr_length: u16,
    bleach: u16,
    weights: Vec<u8>,
}

impl<'a> FromData<'a> for ModelMultipart {
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
        let mut number_of_hashtables = None;
        let mut addr_length = None;
        let mut bleach = None;
        let mut weights = None;

        mp.foreach_entry(|mut entry| match &*entry.headers.name {
            "number_of_hashtables" => {
                let mut t = String::new();
                entry.data.read_to_string(&mut t).expect("not text");
                let n = t.parse::<u16>().expect("not number");
                number_of_hashtables = Some(n);
            }
            "addr_length" => {
                let mut t = String::new();
                entry.data.read_to_string(&mut t).expect("not text");
                let n = t.parse::<u16>().expect("not number");
                addr_length = Some(n);
            }
            "bleach" => {
                let mut t = String::new();
                entry.data.read_to_string(&mut t).expect("not text");
                let n = t.parse::<u16>().expect("not number");
                bleach = Some(n);
            }
            "weights" => {
                let mut d = Vec::new();
                entry.data.read_to_end(&mut d).expect("not weights");
                weights = Some(d);
            }
            other => panic!("No known key {}", other),
        })
        .expect("Unable to iterate");

        let v = ModelMultipart {
            number_of_hashtables: number_of_hashtables.expect("number_of_hashtables not set"),
            addr_length: addr_length.expect("addr_length not set"),
            bleach: bleach.expect("bleach not set"),
            weights: weights.expect("weights not set"),
        };

        // End custom

        Outcome::Success(v)
    }
}

pub struct ClassifyImageMultipart {
    image: Vec<u8>,
}

impl<'a> FromData<'a> for ClassifyImageMultipart {
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
        let mut image = None;

        mp.foreach_entry(|mut entry| match &*entry.headers.name {
            "image" => {
                let mut d = Vec::new();
                entry.data.read_to_end(&mut d).expect("not image");
                image = Some(d);
            }
            other => panic!("No known key {}", other),
        })
        .expect("Unable to iterate");

        let v = ClassifyImageMultipart {
            image: image.expect("image not set"),
        };

        // End custom

        Outcome::Success(v)
    }
}

pub struct TrainImageMultipart {
    label: String,
    image: Vec<u8>,
}

impl<'a> FromData<'a> for TrainImageMultipart {
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
        let mut label = None;
        let mut image = None;

        mp.foreach_entry(|mut entry| match &*entry.headers.name {
            "label" => {
                let mut t = String::new();
                entry.data.read_to_string(&mut t).expect("not text");
                label = Some(t);
            }
            "image" => {
                let mut d = Vec::new();
                entry.data.read_to_end(&mut d).expect("not image");
                image = Some(d);
            }
            other => panic!("No known key {}", other),
        })
        .expect("Unable to iterate");

        let v = TrainImageMultipart {
            label: label.expect("label not set"),
            image: image.expect("image not set"),
        };

        // End custom

        Outcome::Success(v)
    }
}
