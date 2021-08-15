use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;

#[derive(Debug)]
pub struct MnistData {
    pub sizes: Vec<i32>,
    pub data: Vec<u8>,
}

impl MnistData {
    pub fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut contents: Vec<u8> = Vec::new();
        let mut gz = flate2::read::GzDecoder::new(f);
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

pub fn load_mnist(folder: &str, prefix: &str) -> Result<(Vec<Vec<u8>>, Vec<u8>), std::io::Error> {
    let label_data =
        MnistData::new(&(File::open(folder.to_owned() + prefix + "-labels-idx1-ubyte.gz"))?)?;
    let images_data =
        MnistData::new(&(File::open(folder.to_owned() + prefix + "-images-idx3-ubyte.gz"))?)?;
    let mut images: Vec<Vec<u8>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        images.push(image_data);
    }

    let classifications: Vec<u8> = label_data.data.clone();
    Ok((images, classifications))
}
