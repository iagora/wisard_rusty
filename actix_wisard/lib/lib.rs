extern crate wisard;

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
pub async fn run() -> std::io::Result<()> {
    let wis = web::Data::new(RwLock::new(wisard::dict_wisard::Wisard::<u8>::new()));

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    HttpServer::new(move || {
        App::new()
            .app_data(wis.clone())
            .wrap(middleware::Compress::default())
            .wrap(middleware::Logger::new(
                "%a %t %r %b %{Referer}i %{User-Agent}i %s %T",
            ))
            .service(web::resource("/new").route(web::post().to(new)))
            .service(web::resource("/train?{label}>").route(web::post().to(train)))
            .service(web::resource("/classify").route(web::post().to(classify)))
            .service(web::resource("/info").route(web::get().to(info)))
            .service(
                web::resource("/model")
                    .route(web::get().to(save))
                    .route(
                        web::post()
                            .guard(guard::Header("content-encoding", "gzip"))
                            .to(load),
                    )
                    .route(web::delete().to(erase)),
            )
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn new(
    wis: web::Data<RwLock<wisard::dict_wisard::Wisard<u8>>>,
    web::Query(model_info): web::Query<ModelInfo>,
) -> Result<HttpResponse, Error> {
    let mut unlocked_wis = match wis.write() {
        Ok(unlocked_wis) => unlocked_wis,
        Err(error) => {
            return Ok(HttpResponse::from_error(error::ErrorInternalServerError(
                format!("Failed to get lock on cache: {}", error),
            )))
        }
    };
    unlocked_wis.erase_and_change_hyperparameters(
        model_info.hashtables,
        model_info.addresses,
        model_info.bleach,
    );
    Ok(HttpResponse::Ok().into())
}

async fn info(
    wis: web::Data<RwLock<wisard::dict_wisard::Wisard<u8>>>,
) -> Result<HttpResponse, Error> {
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

const STREAM_MAX_SIZE: usize = 10_000_000; // 500MB limit

async fn train(
    wis: web::Data<RwLock<wisard::dict_wisard::Wisard<u8>>>,
    web::Path(label): web::Path<String>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error> {
    let mut v = Vec::new();
    while let Some(chunk) = payload.next().await {
        let data = chunk.unwrap();
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
    unlocked_wis.train(v, label);

    Ok(HttpResponse::Ok().into())
}

async fn classify(
    wis: web::Data<RwLock<wisard::dict_wisard::Wisard<u8>>>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error> {
    let mut v = Vec::new();
    while let Some(chunk) = payload.next().await {
        let data = chunk.unwrap();
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
    let (label, _, _) = unlocked_wis.classify(v);
    Ok(HttpResponse::Ok().json(ClassifyResponse { label: label }))
}

async fn save(
    wis: web::Data<RwLock<wisard::dict_wisard::Wisard<u8>>>,
) -> Result<HttpResponse, Error> {
    Ok(HttpResponse::Ok()
        .encoding(ContentEncoding::Gzip)
        .body(wis.read().unwrap().save()))
}

const WEIGHT_MAX_SIZE: usize = 500_000_000; // 500MB limit

async fn load(
    wis: web::Data<RwLock<wisard::dict_wisard::Wisard<u8>>>,
    mut payload: web::Payload,
) -> Result<HttpResponse, Error> {
    let mut decoder = Decompress::new(&mut payload, ContentEncoding::Gzip);
    let mut v = Vec::new();
    while let Some(chunk) = decoder.next().await {
        let data = chunk.unwrap();
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
    unlocked_wis.load(&v);

    Ok(HttpResponse::Ok().into())
}

async fn erase(
    wis: web::Data<RwLock<wisard::dict_wisard::Wisard<u8>>>,
) -> Result<HttpResponse, Error> {
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
