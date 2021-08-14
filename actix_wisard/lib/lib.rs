extern crate wisard;

use wisard::wisard_traits::WisardNetwork;
// use actix_multipart::Multipart;
use actix_web::{
    dev::BodyEncoding, dev::Decompress, error, guard, http::ContentEncoding, middleware, web, App,
    Error, HttpResponse, HttpServer,
};
use async_std::prelude::*;
use env_logger::Env;
use futures::StreamExt; //, TryStreamExt};
use image::imageops;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

#[actix_web::main]
pub async fn run<T, K>(wis: T) -> std::io::Result<()>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync + From<u8> + 'static,
{
    let wis = web::Data::new(RwLock::new(wis));

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    HttpServer::new(move || {
        App::new()
            .app_data(wis.clone())
            .wrap(middleware::Compress::default())
            .wrap(middleware::Logger::new(
                "%a %t %r %b %{Referer}i %{User-Agent}i %s %T",
            ))
            .service(web::resource("/new").route(web::post().to(new::<T, K>)))
            .service(web::resource("/train").route(web::post().to(train::<T, K>)))
            .service(web::resource("/classify").route(web::post().to(classify::<T, K>)))
            .service(web::resource("/info").route(web::get().to(info::<T, K>)))
            .service(
                web::resource("/model")
                    .route(web::get().to(save::<T, K>))
                    .route(
                        web::post()
                            .guard(guard::Header("content-encoding", "gzip"))
                            .to(load::<T, K>),
                    )
                    .route(web::delete().to(erase::<T, K>)),
            )
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

async fn new<T, K>(
    wis: web::Data<RwLock<T>>,
    web::Json(model_info): web::Json<ModelInfoRequest>,
) -> Result<HttpResponse, Error>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync,
{
    let mut unlocked_wis = match wis.write() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };

    let temp_target_size = match model_info.target_size {
        Some(t) => t,
        None => (64, 64),
    };

    if (model_info.hashtables as u32 * model_info.addresses as u32)
        > (temp_target_size.0 * temp_target_size.1)
    {
        return Ok(HttpResponse::from_error(error::ErrorBadRequest(format!(
            "The image resize dimensions can't be smaller than the sampling range"
        ))));
    }

    if let Some(map) = &model_info.mapping {
        if (model_info.hashtables as usize * model_info.addresses as usize) > map.len() {
            return Ok(HttpResponse::from_error(error::ErrorBadRequest(format!(
                "The mapping size must be bigger than (number_of_hashtables * addresses_length)"
            ))));
        }
        if (temp_target_size.0 as usize * temp_target_size.1 as usize) < map.len() {
            return Ok(HttpResponse::from_error(error::ErrorBadRequest(format!(
                "The image resize dimensions can't be smaller than the mapping"
            ))));
        }
    }

    unlocked_wis.change_hyperparameters(
        model_info.hashtables,
        model_info.addresses,
        model_info.bleach,
        model_info.target_size,
        model_info.mapping,
    );
    Ok(HttpResponse::Ok().into())
}

async fn info<T, K>(wis: web::Data<RwLock<T>>) -> Result<HttpResponse, Error>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync,
{
    let unlocked_wis = match wis.read() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };

    let (hashtables, addresses, bleach, target_size, mapping) = unlocked_wis.get_info();
    Ok(HttpResponse::Ok().json(ModelInfoResponse {
        hashtables: hashtables,
        addresses: addresses,
        bleach: bleach,
        target_size: target_size,
        mapping: mapping,
    }))
}

const STREAM_MAX_SIZE: usize = 10_000_000; // 10MB limit

async fn train<T, K>(
    wis: web::Data<RwLock<T>>,
    web::Query(label_r): web::Query<ClassifyRequest>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync + From<u8>,
{
    let mut v = Vec::new();
    while let Some(chunk) = payload.next().await {
        let data = chunk?;
        // limit max size of in-memory payload
        if (v.len() + data.len()) > STREAM_MAX_SIZE {
            return Err(error::ErrorBadRequest("overflow"));
        }
        v.write_all(&data).await?;
    }

    let img = match image::load_from_memory(&v) {
        Ok(img) => img,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get image: {}", error),
            )))
        }
    };

    let mut unlocked_wis = match wis.write() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };

    let target_size = unlocked_wis.target_size();

    match unlocked_wis.train(
        img.grayscale()
            .resize_exact(target_size.0, target_size.1, imageops::Nearest)
            .as_bytes()
            .iter()
            .map(|x| K::from(*x))
            .collect(),
        label_r.label,
    ) {
        Ok(_) => return Ok(HttpResponse::Ok().into()),
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Wisard internal error: {}", error),
            )))
        }
    }
}

async fn classify<T, K>(
    wis: web::Data<RwLock<T>>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync + From<u8>,
{
    let mut v = Vec::new();
    while let Some(chunk) = payload.next().await {
        let data = chunk?;
        // limit max size of in-memory payload
        if (v.len() + data.len()) > STREAM_MAX_SIZE {
            return Err(error::ErrorBadRequest("overflow"));
        }
        v.write_all(&data).await?;
    }

    let img = match image::load_from_memory(&v) {
        Ok(img) => img,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get image: {}", error),
            )))
        }
    };

    let unlocked_wis = match wis.read() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };

    let target_size = unlocked_wis.target_size();

    match unlocked_wis.classify(
        img.grayscale()
            .resize_exact(target_size.0, target_size.1, imageops::Nearest)
            .as_bytes()
            .iter()
            .map(|x| K::from(*x))
            .collect(),
    ) {
        Ok(label) => return Ok(HttpResponse::Ok().json(ClassifyResponse { label: label })),
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Wisard internal error: {}", error),
            )))
        }
    }
}

async fn save<T, K>(wis: web::Data<RwLock<T>>) -> Result<HttpResponse, Error>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync,
{
    let unlocked_wis = match wis.read() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };
    let encoded = match unlocked_wis.save() {
        Ok(e) => e,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Wisard found an error while saving: {}", error),
            )))
        }
    };

    Ok(HttpResponse::Ok()
        .encoding(ContentEncoding::Gzip)
        .body(encoded))
}

const WEIGHT_MAX_SIZE: usize = 500_000_000; // 500MB limit

async fn load<T, K>(
    wis: web::Data<RwLock<T>>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync,
{
    let mut decoder = Decompress::new(&mut payload, ContentEncoding::Gzip);
    let mut v = Vec::new();
    while let Some(chunk) = decoder.next().await {
        let data = chunk?;
        // limit max size of in-memory payload
        if (v.len() + data.len()) > WEIGHT_MAX_SIZE {
            return Err(error::ErrorBadRequest("overflow"));
        }
        v.write_all(&data).await?;
    }

    let mut unlocked_wis = match wis.write() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };

    match unlocked_wis.load(&v) {
        Ok(_) => return Ok(HttpResponse::Ok().into()),
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Wisard internal error: {}", error),
            )))
        }
    }
}

async fn erase<T, K>(wis: web::Data<RwLock<T>>) -> Result<HttpResponse, Error>
where
    T: WisardNetwork<K> + Send + Sync + 'static,
    K: PartialOrd + Copy + Send + Sync,
{
    let mut unlocked_wis = match wis.write() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };
    unlocked_wis.erase();

    Ok(HttpResponse::Ok().into())
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfoRequest {
    hashtables: u16,
    addresses: u16,
    bleach: u16,
    target_size: Option<(u32, u32)>,
    mapping: Option<Vec<u64>>,
}

impl Default for ModelInfoRequest {
    fn default() -> Self {
        ModelInfoRequest {
            hashtables: 35,
            addresses: 21,
            bleach: 0,
            target_size: None,
            mapping: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfoResponse {
    hashtables: u16,
    addresses: u16,
    bleach: u16,
    target_size: (u32, u32),
    mapping: Vec<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ClassifyResponse {
    label: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ClassifyRequest {
    label: String,
}
