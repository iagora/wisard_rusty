extern crate dict_wisard;

use actix_files::NamedFile;
use actix_multipart::Multipart;
use actix_web::{middleware, web, App, Error, HttpResponse, HttpServer};
use async_std::prelude::*;
use env_logger::Env;
use futures::{StreamExt, TryStreamExt};
use serde::Deserialize;
use std::sync::Mutex;

#[actix_web::main]
pub async fn run() -> std::io::Result<()> {
    let wis = web::Data::new(Mutex::new(dict_wisard::wisard::Wisard::<u8>::new()));

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .wrap(middleware::Logger::new("%a %{User-Agent}i"))
            .wrap(middleware::Compress::default())
            .app_data(wis.clone())
            .service(web::resource("/new").route(web::post().to(new)))
            .service(web::resource("/with_model").route(web::post().to(with_model)))
            .service(web::resource("/train?<label>").route(web::post().to(train)))
            .service(web::resource("/classify").route(web::post().to(classify)))
            .service(
                web::resource("/model")
                    .route(web::get().to(save))
                    .route(web::post().to(load))
                    .route(web::delete().to(erase)),
            )
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn new(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
    web::Query(model_info): web::Query<ModelRequestType>,
) -> Result<HttpResponse, Error> {
    wis.lock().unwrap().erase_and_change_hyperparameters(
        model_info.hashtables,
        model_info.addresses,
        model_info.bleach,
    );
    Ok(HttpResponse::Created().into())
}

async fn with_model(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
    web::Query(model_info): web::Query<ModelRequestType>,
    mut payload: Multipart,
) -> Result<HttpResponse, Error> {
    let mut v = Vec::new();
    // iterate over multipart stream
    while let Ok(Some(mut field)) = payload.try_next().await {
        // let content_type = field
        //     .content_disposition()
        //     .ok_or_else(|| actix_web::error::ParseError::Incomplete)?;
        // let filename = content_type
        //     .get_filename()
        //     .ok_or_else(|| actix_web::error::ParseError::Incomplete)?;

        // Field in turn is stream of *Bytes* object
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            v.write_all(&data).await?;
        }
    }
    let mut unlocked_wis = wis.lock().unwrap();
    unlocked_wis.erase_and_change_hyperparameters(
        model_info.hashtables,
        model_info.addresses,
        model_info.bleach,
    );
    unlocked_wis.load(&v);

    Ok(HttpResponse::Created().into())
}

async fn train(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
    web::Path(label): web::Path<String>,
    mut payload: Multipart,
) -> Result<HttpResponse, Error> {
    let mut v = Vec::new();
    // iterate over multipart stream
    while let Ok(Some(mut field)) = payload.try_next().await {
        // Field in turn is stream of *Bytes* object
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            v.write_all(&data).await?;
        }
    }
    let mut unlocked_wis = wis.lock().unwrap();
    unlocked_wis.train(v, label);

    Ok(HttpResponse::Ok().into())
}

async fn classify(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
    mut payload: Multipart,
) -> Result<HttpResponse, Error> {
    let mut v = Vec::new();
    // iterate over multipart stream
    while let Ok(Some(mut field)) = payload.try_next().await {
        // Field in turn is stream of *Bytes* object
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            v.write_all(&data).await?;
        }
    }
    let (label, _, _) = wis.lock().unwrap().classify(v);
    Ok(HttpResponse::Ok()
        .content_type("text/plain")
        .body(format!("{}", label)))
}

async fn save(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
) -> actix_web::Result<NamedFile, Error> {
    //Cursor<Vec<u8>>
    wis.lock().unwrap().save_to_file("/tmp/weights.bin");
    Ok(NamedFile::open("/tmp/weights.bin")?)
}

async fn load(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
    mut payload: Multipart,
) -> Result<HttpResponse, Error> {
    let mut v = Vec::new();
    // iterate over multipart stream
    while let Ok(Some(mut field)) = payload.try_next().await {
        // Field in turn is stream of *Bytes* object
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            v.write_all(&data).await?;
        }
    }

    wis.lock().unwrap().load(&v);

    Ok(HttpResponse::Ok().into())
}

async fn erase(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
) -> Result<HttpResponse, Error> {
    wis.lock().unwrap().erase();

    Ok(HttpResponse::Ok().into())
}

#[derive(Debug, Deserialize)]
struct ModelRequestType {
    hashtables: u16,
    addresses: u16,
    bleach: u16,
}
