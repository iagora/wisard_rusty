use dataloaders::mnist::load_mnist;
use http_types::Body;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[async_std::main]
async fn main() -> surf::Result<()> {
    println!("Rusty WiSARD Client - MNIST 🦀🦀🦀");

    let wis_info: ModelInfoResponse = surf::get("http://localhost:8080/info").recv_json().await?;

    println!("Number of hashtables: {}", wis_info.hashtables);
    println!("Address size: {}", wis_info.addresses);
    println!("Bleaching: {}", wis_info.bleach);

    println!("\n-----------------\nTraining\n-----------------");
    let now = Instant::now();

    let (images, classifications) = load_mnist("data/mnist/", "train")?;

    println!("Training data has {} images", classifications.len());
    println!(
        "Parsing the training dataset took: {} milliseconds",
        now.elapsed().as_millis()
    );

    let now = Instant::now();
    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        let label = ClassifyRequest {
            label: classification.to_string(),
        };
        surf::post("http://localhost:8080/train")
            .body(image)
            .query(&label)?
            .send()
            .await?;
    }
    println!("Training took: {} milliseconds", now.elapsed().as_millis());

    println!("-----------------\nTesting\n-----------------");
    let now = Instant::now();
    let (images, classifications) = load_mnist("data/mnist/", "t10k")?;

    println!("Testing data has {} images", classifications.len());
    println!(
        "Parsing the test dataset took: {} milliseconds",
        now.elapsed().as_millis()
    );
    let mut hit: u64 = 0;
    let mut count: u64 = 0;

    let now = Instant::now();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        // let tuple: (String, f64, f64);
        let ClassifyResponse { label } = surf::post("http://localhost:8080/classify")
            .body(Body::from_bytes(image))
            .recv_json()
            .await?;
        // tuple = wis.classify(image);
        if label == classification.to_string() {
            hit += 1;
        }
        count += 1;
    }
    println!("Testing took: {} milliseconds", now.elapsed().as_millis());

    println!("Accuracy: {:.4}", hit as f64 / count as f64);

    Ok(())
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
