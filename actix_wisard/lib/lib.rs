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
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

#[actix_web::main]
pub async fn run<K>(wis: Box<dyn WisardNetwork<K> + Send + Sync + 'static>) -> std::io::Result<()>
where
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
            .service(web::resource("/new").route(web::post().to(new::<K>)))
            .service(web::resource("/train?{label}>").route(web::post().to(train::<K>)))
            .service(web::resource("/classify").route(web::post().to(classify::<K>)))
            .service(web::resource("/info").route(web::get().to(info::<K>)))
            .service(
                web::resource("/model")
                    .route(web::get().to(save::<K>))
                    .route(
                        web::post()
                            .guard(guard::Header("content-encoding", "gzip"))
                            .to(load::<K>),
                    )
                    .route(web::delete().to(erase::<K>)),
            )
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn new<K>(
    wis: web::Data<RwLock<Box<dyn WisardNetwork<K> + Send + Sync + 'static>>>,
    web::Query(model_info): web::Query<ModelInfo>,
) -> Result<HttpResponse, Error>
where
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
    unlocked_wis.change_hyperparameters(
        model_info.hashtables,
        model_info.addresses,
        model_info.bleach,
    );
    Ok(HttpResponse::Ok().into())
}

async fn info<K>(
    wis: web::Data<RwLock<Box<dyn WisardNetwork<K> + Send + Sync + 'static>>>,
) -> Result<HttpResponse, Error>
where
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
    let (hashtables, addresses, bleach) = unlocked_wis.get_info();
    Ok(HttpResponse::Ok().json(ModelInfo {
        hashtables: hashtables,
        addresses: addresses,
        bleach: bleach,
    }))
}

const STREAM_MAX_SIZE: usize = 10_000_000; // 10MB limit

async fn train<K>(
    wis: web::Data<RwLock<Box<dyn WisardNetwork<K> + Send + Sync + 'static>>>,
    web::Path(label): web::Path<String>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error>
where
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
    let mut unlocked_wis = match wis.write() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };

    match unlocked_wis.train(v.iter().map(|x| K::from(*x)).collect(), label) {
        Ok(_) => return Ok(HttpResponse::Ok().into()),
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Wisard internal error: {}", error),
            )))
        }
    }
}

async fn classify<K>(
    wis: web::Data<RwLock<Box<dyn WisardNetwork<K> + Send + Sync + 'static>>>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error>
where
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

    let unlocked_wis = match wis.read() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };

    match unlocked_wis.classify(v.iter().map(|x| K::from(*x)).collect()) {
        Ok(label) => return Ok(HttpResponse::Ok().json(ClassifyResponse { label: label })),
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Wisard internal error: {}", error),
            )))
        }
    }
}

async fn save<K>(
    wis: web::Data<RwLock<Box<dyn WisardNetwork<K> + Send + Sync + 'static>>>,
) -> Result<HttpResponse, Error>
where
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

async fn load<K>(
    wis: web::Data<RwLock<Box<dyn WisardNetwork<K> + Send + Sync + 'static>>>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error>
where
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

async fn erase<K>(
    wis: web::Data<RwLock<Box<dyn WisardNetwork<K> + Send + Sync + 'static>>>,
) -> Result<HttpResponse, Error>
where
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

#[derive(Default, Debug, Deserialize, Serialize)]
struct ModelInfo {
    hashtables: u16,
    addresses: u16,
    bleach: u16,
}

#[derive(Debug, Deserialize, Serialize)]
struct ClassifyResponse {
    label: String,
}
