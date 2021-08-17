use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
use std::path::Path;

use crate::error::WisardError;
use crate::wisard_traits::WisardNetwork;

#[derive(Deserialize, Serialize, Debug)]
pub struct Discriminator {
    number_of_hashtables: u16,
    h_rams: Vec<HashMap<u64, u16>>,
    times_trained: u64,
}

impl Discriminator {
    pub fn new(num: u16) -> Discriminator {
        Discriminator {
            number_of_hashtables: num,
            h_rams: vec![HashMap::new(); num as usize],
            times_trained: 0,
        }
    }

    pub fn train(&mut self, x: Vec<u64>) -> Result<(), WisardError> {
        for i in 0..self.number_of_hashtables {
            let key = x[i as usize];
            let counter = self
                .h_rams
                .get_mut(i as usize)
                .ok_or_else(|| WisardError::WisardOutOfBounds)?
                .entry(key)
                .or_insert(0);
            *counter += 1;
        }
        self.times_trained += 1;
        Ok(())
    }

    pub fn classify(&self, x: &Vec<u64>, bleach: u16) -> (u64, u64) {
        let mut votes: u64 = 0;
        for i in 0..self.number_of_hashtables {
            let key = x[i as usize];
            if let Some(x) = self.h_rams[i as usize].get(&key) {
                if *x > bleach {
                    votes += 1
                };
            }
        }
        (votes, self.times_trained)
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Wisard<T> {
    discs: HashMap<String, Discriminator>,
    addr_length: u16,
    number_of_hashtables: u16,
    mapping: Vec<u64>,
    last_rank: u64,
    rank_tables: HashMap<Vec<u64>, u64>,
    bleach: u16,
    target_size: (u32, u32),
    phantom: PhantomData<T>,
}

impl<T> WisardNetwork<T> for Wisard<T> {
    fn train(&mut self, data: Vec<T>, label: String) -> Result<(), WisardError>
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        if !self.discs.contains_key(&label) {
            self.discs
                .insert(label.clone(), Discriminator::new(self.number_of_hashtables));
        }

        let samples = self.mapping.clone();
        let samples = samples
            .iter()
            .map(|&i| data.get(i as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| WisardError::WisardOutOfBounds)?;
        let addresses: Vec<u64> = self.ranks_t(samples);
        let disc = self.discs.get_mut(&label).unwrap();
        disc.train(addresses)?;
        Ok(())
    }
    fn classify(&self, data: Vec<T>) -> Result<String, WisardError>
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        let samples = self.mapping.clone();
        let samples = samples
            .iter()
            .map(|&i| data.get(i as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| WisardError::WisardOutOfBounds)?;
        let addresses: Vec<u64> = self.ranks_c(samples);
        let discs = &self.discs;
        let mut votes: Vec<(String, (u64, u64))> = discs
            .iter()
            .map(|d| (d.0.to_string(), d.1.classify(&addresses, self.bleach)))
            .collect();
        votes.sort_by(|a, b| (a.1).0.partial_cmp(&(b.1).0).unwrap());

        let biggest = votes
            .iter()
            .last()
            .ok_or_else(|| WisardError::WisardOutOfBounds)?;

        Ok(
            biggest.0.clone(), // elected label
        )
    }
    fn save(&self) -> Result<Vec<u8>, WisardError> {
        let encoded: Vec<u8> = match bincode::serialize(&self) {
            Ok(enc) => enc,
            Err(_) => {
                return Err(WisardError::WisardValidationFailed(String::from(
                    "Could not serialize wisard!",
                )))
            }
        };
        Ok(encoded)
    }
    fn load(&mut self, stream: &[u8]) -> Result<(), WisardError> {
        let decoded: Wisard<T> = match bincode::deserialize(stream) {
            Ok(res) => res,
            Err(_) => {
                return Err(WisardError::WisardValidationFailed(String::from(
                    "Could not deserialize with bincode!",
                )))
            }
        };
        self.discs = decoded.discs;
        self.addr_length = decoded.addr_length;
        self.number_of_hashtables = decoded.number_of_hashtables;
        self.mapping = decoded.mapping;
        self.last_rank = decoded.last_rank;
        self.rank_tables = decoded.rank_tables;
        self.target_size = decoded.target_size;
        self.bleach = decoded.bleach;
        Ok(())
    }
    fn erase(&mut self) {
        self.number_of_hashtables = 35;
        self.addr_length = 21;
        self.bleach = 0;
        self.mapping.shuffle(&mut thread_rng());
        self.discs = HashMap::new();
        self.last_rank = 0;
        self.target_size = (28, 28);
        self.rank_tables = HashMap::new()
    }
    fn change_hyperparameters(
        &mut self,
        number_of_hashtables: u16,
        addr_length: u16,
        bleach: u16,
        target_size: Option<(u32, u32)>,
        mapping: Option<Vec<u64>>,
    ) -> Result<(), WisardError> {
        // randomizes the mapping
        let mapping = match mapping {
            Some(map) => map,
            None => {
                let mut rng_mapping =
                    (0..addr_length as u64 * number_of_hashtables as u64).collect::<Vec<u64>>();
                rng_mapping.shuffle(&mut thread_rng());
                rng_mapping
            }
        };
        let target_size = match target_size {
            Some(t) => t,
            None => (28, 28),
        };

        if (number_of_hashtables as u32 * addr_length as u32) > (target_size.0 * target_size.1) {
            return Err(WisardError::WisardValidationFailed(String::from(
                "The image resize dimensions can't be smaller than the sampling range",
            )));
        }

        if (number_of_hashtables as usize * addr_length as usize) > mapping.len() {
            return Err(WisardError::WisardValidationFailed(String::from(
                "The mapping size must be bigger than (number_of_hashtables * addresses_length)",
            )));
        }
        if (target_size.0 as usize * target_size.1 as usize) < mapping.len() {
            return Err(WisardError::WisardValidationFailed(String::from(
                "The image resize dimensions can't be smaller than the mapping",
            )));
        }

        self.discs = HashMap::new();
        self.addr_length = addr_length;
        self.number_of_hashtables = number_of_hashtables;
        self.mapping = mapping;
        self.last_rank = 0;
        self.rank_tables = HashMap::new();
        self.target_size = target_size;
        self.bleach = bleach;

        return Ok(());
    }
    fn get_info(&self) -> (u16, u16, u16, (u32, u32), Vec<u64>) {
        return (
            self.number_of_hashtables,
            self.addr_length,
            self.bleach,
            self.target_size,
            self.mapping.clone(),
        );
    }
    fn target_size(&self) -> (u32, u32) {
        return self.target_size;
    }
}

impl<T> Wisard<T> {
    pub fn new() -> Self
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        Wisard::with_params(35, 21, 0, None, None).unwrap()
    }

    pub fn with_params(
        number_of_hashtables: u16,
        addr_length: u16,
        bleach: u16,
        target_size: Option<(u32, u32)>,
        mapping: Option<Vec<u64>>,
    ) -> Result<Self, WisardError> {
        // randomizes the mapping
        let mapping = match mapping {
            Some(map) => map,
            None => {
                let mut rng_mapping =
                    (0..addr_length as u64 * number_of_hashtables as u64).collect::<Vec<u64>>();
                rng_mapping.shuffle(&mut thread_rng());
                rng_mapping
            }
        };

        let target_size = match target_size {
            Some(t) => t,
            None => (28, 28),
        };

        if (number_of_hashtables as u32 * addr_length as u32) > (target_size.0 * target_size.1) {
            return Err(WisardError::WisardValidationFailed(String::from(
                "The image resize dimensions can't be smaller than the sampling range",
            )));
        }

        if (number_of_hashtables as usize * addr_length as usize) > mapping.len() {
            return Err(WisardError::WisardValidationFailed(String::from(
                "The mapping size must be bigger than (number_of_hashtables * addresses_length)",
            )));
        }
        if (target_size.0 as usize * target_size.1 as usize) < mapping.len() {
            return Err(WisardError::WisardValidationFailed(String::from(
                "The image resize dimensions can't be smaller than the mapping",
            )));
        }

        Ok(Wisard::<T> {
            discs: HashMap::new(),
            addr_length: addr_length,
            number_of_hashtables: number_of_hashtables,
            mapping: mapping,
            last_rank: 0,
            rank_tables: HashMap::new(),
            bleach: bleach,
            target_size: target_size,
            phantom: PhantomData,
        })
    }

    fn ranks_t(&mut self, samples: Vec<&T>) -> Vec<u64>
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        let mut vetor = Vec::with_capacity(self.addr_length as usize);
        let mut addresses = Vec::new();
        for i in (0..samples.len()).step_by(self.addr_length as usize) {
            if i + self.addr_length as usize <= samples.len() {
                vetor.append(&mut samples[i..i + self.addr_length as usize].to_vec());
            } else {
                vetor.append(&mut samples[i..samples.len()].to_vec());
            }
            let mut tuples: Vec<(u64, &T)> = vetor
                .iter()
                .enumerate()
                .map(|x| (x.0 as u64, *x.1))
                .collect();
            tuples.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap()); // TODO: treat the Option
            let address: Vec<u64> = tuples.iter().map(|a| a.0).collect();
            if !self.rank_tables.contains_key(&address) {
                self.rank_tables.insert(address.clone(), self.last_rank);
                self.last_rank += 1;
            }
            addresses.push(*self.rank_tables.get(&address).unwrap()); // TODO: treat the Option
            vetor.clear();
        }
        addresses
    }

    fn ranks_c(&self, samples: Vec<&T>) -> Vec<u64>
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        let mut vetor = Vec::with_capacity(self.addr_length as usize);
        let mut addresses = Vec::new();
        for i in (0..samples.len()).step_by(self.addr_length as usize) {
            if i + self.addr_length as usize <= samples.len() {
                vetor.append(&mut samples[i..i + self.addr_length as usize].to_vec());
            } else {
                vetor.append(&mut samples[i..samples.len()].to_vec());
            }
            let mut tuples: Vec<(u64, &T)> = vetor
                .iter()
                .enumerate()
                .map(|x| (x.0 as u64, *x.1))
                .collect();
            tuples.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap()); // TODO: treat the Option
            let address: Vec<u64> = tuples.iter().map(|a| a.0).collect();
            if !self.rank_tables.contains_key(&address) {
                // self.rank_tables.insert(address.clone(), self.last_rank);
                let tmp_rank = 1 + self.last_rank;
                addresses.push(tmp_rank);
            } else {
                addresses.push(*self.rank_tables.get(&address).unwrap()); // TODO: treat the Option
            }
            vetor.clear();
        }
        addresses
    }
    pub fn save_to_file(&self, path: &Path) -> Result<(), WisardError> {
        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(_) => return Err(WisardError::WisardIOError),
        };
        match bincode::serialize_into(&mut file, &self) {
            Ok(_) => Ok(()),
            Err(_) => {
                return Err(WisardError::WisardValidationFailed(String::from(
                    "Couldn't serialize wisard into a file!",
                )))
            }
        }
    }
    pub fn load_from_file(&mut self, path: &Path) -> Result<(), WisardError> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Err(WisardError::WisardIOError),
        };
        let decoded: Wisard<T> = match bincode::deserialize_from(file) {
            Ok(d) => d,
            Err(_) => {
                return Err(WisardError::WisardValidationFailed(String::from(
                    "Could not deserialize the file into a wisard!",
                )))
            }
        };
        self.discs = decoded.discs;
        self.addr_length = decoded.addr_length;
        self.number_of_hashtables = decoded.number_of_hashtables;
        self.mapping = decoded.mapping;
        self.last_rank = decoded.last_rank;
        self.rank_tables = decoded.rank_tables;
        self.bleach = decoded.bleach;
        Ok(())
    }
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_lib_ranks() {
        // this test verifies that ranks is able to push address to rank_tables
        let mut wis = Wisard::new();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks_t(samples.iter().collect());
        assert!(!wis.rank_tables.is_empty());
    }
    #[test]
    fn test_lib_rank_table_length() {
        // this test ensures that the same addresses aren't pushed into the rank_tables
        // repeatedly
        let mut wis = Wisard::new();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks_t(samples.iter().collect());
        let length1 = wis.rank_tables.len();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks_t(samples.iter().collect());
        let length2 = wis.rank_tables.len();
        assert_eq!(length1, length2);
    }

    #[test]
    fn test_lib_rank_addresses() {
        // this test verifies that for each new piece of data, a correct rank is attributed
        let mut wis = Wisard::new();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let addresses = wis.ranks_t(samples.iter().collect());
        assert_eq!(addresses, vec![0, 1, 2]);
    }
    #[test]
    fn test_lib_rank_different_addresses() {
        // this test verifies that small changes in data get close addresses
        let mut wis = Wisard::new();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let addresses = wis.ranks_t(samples.iter().collect());
        assert_eq!(addresses, vec![0, 1, 2]);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 205,
        ];
        let addresses = wis.ranks_t(samples.iter().collect());
        assert_eq!(addresses, vec![0, 1, 3]);
    }

    #[test]
    fn test_save_load() {
        use std::fs;
        fs::create_dir_all("weights/").unwrap();

        let mut wis = Wisard::new();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let _ = wis.ranks_t(samples.iter().collect());

        wis.save_to_file(Path::new("weights/weigths_u8.bin"))
            .unwrap();

        let mut decoded = Wisard::new();
        decoded
            .load_from_file(Path::new("weights/weigths_u8.bin"))
            .unwrap();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 205,
        ];
        let decoded_addresses = decoded.ranks_t(samples.iter().collect());

        println!("{:?}", decoded_addresses);

        fs::remove_file("weights/weigths_u8.bin").unwrap();
        fs::remove_dir_all("weights/").unwrap();

        assert_eq!(vec![0, 1, 3], decoded_addresses);
    }

    #[test]
    fn test_erase() {
        let mut wis = Wisard::new();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let _ = wis.ranks_t(samples.iter().collect());

        wis.erase();

        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 205,
        ];
        let decoded_addresses = wis.ranks_t(samples.iter().collect());

        assert_eq!(vec![0, 1, 2], decoded_addresses);
    }
}
