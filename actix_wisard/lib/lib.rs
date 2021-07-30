extern crate dict_wisard;

use actix_files::NamedFile;
use actix_multipart::Multipart;
use actix_web::{middleware, web, App, Error, HttpResponse, HttpServer};
use async_std::prelude::*;
use futures::{StreamExt, TryStreamExt};
use std::sync::Mutex;

#[actix_web::main]
pub async fn run() -> std::io::Result<()> {
    let wis = web::Data::new(Mutex::new(dict_wisard::wisard::Wisard::<u8>::new()));
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Compress::default())
            .service(
                web::resource("/new?<hashtables>&<addresses>&<bleach>").route(web::post().to(new)),
            )
            .service(
                web::resource("/with_model?<hashtables>&<addresses>&<bleach>")
                    .route(web::post().to(with_model)),
            )
            .service(web::resource("/train?<label>").route(web::post().to(train)))
            .service(web::resource("/classify").route(web::post().to(classify)))
            .service(
                web::resource("/model")
                    .route(web::get().to(save))
                    .route(web::post().to(load))
                    .route(web::delete().to(erase)),
            )
            .app_data(wis.clone())
    })
    .bind("127.0.0.1:8080")
    .unwrap()
    .run()
    .await
}

async fn new(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
    web::Path((hashtables, addresses, bleach)): web::Path<(u16, u16, u16)>,
) -> Result<HttpResponse, Error> {
    wis.lock()
        .unwrap()
        .erase_and_change_hyperparameters(hashtables, addresses, bleach);
    Ok(HttpResponse::Created().into())
}

async fn with_model(
    wis: web::Data<Mutex<dict_wisard::wisard::Wisard<u8>>>,
    web::Path((hashtables, addresses, bleach)): web::Path<(u16, u16, u16)>,
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
    unlocked_wis.erase_and_change_hyperparameters(hashtables, addresses, bleach);
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
