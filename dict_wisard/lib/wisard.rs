use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
use std::path::Path;

#[derive(Deserialize, Serialize)]
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

    pub fn train(&mut self, x: Vec<u64>) {
        for i in 0..self.number_of_hashtables {
            let key = x[i as usize];
            let counter = self
                .h_rams
                .get_mut(i as usize)
                .unwrap()
                .entry(key)
                .or_insert(0);
            *counter += 1;
        }
        self.times_trained += 1;
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

#[derive(Deserialize, Serialize)]
pub struct Wisard<T> {
    discs: HashMap<String, Discriminator>,
    addr_length: u16,
    number_of_hashtables: u16,
    mapping: Vec<u64>,
    last_rank: u64,
    rank_tables: HashMap<Vec<u64>, u64>,
    bleach: u16,
    phantom: PhantomData<T>,
}

impl<T> Wisard<T> {
    pub fn new<O1, O2, O3>(number_of_hashtables: O1, addr_length: O2, bleach: O3) -> Self
    where
        T: PartialOrd + Copy,
        O1: Into<Option<u16>>,
        O2: Into<Option<u16>>,
        O3: Into<Option<u16>>,
    {
        let mut number_of_hashtables_: u16 = 28;
        match number_of_hashtables.into() {
            Some(x) => {
                number_of_hashtables_ = x;
            }
            _ => {}
        }
        let mut addr_length_: u16 = 28;
        match addr_length.into() {
            Some(x) => {
                addr_length_ = x;
            }
            _ => {}
        }
        let mut bleach_: u16 = 0;
        match bleach.into() {
            Some(x) => {
                bleach_ = x;
            }
            _ => {}
        }
        // randomizes the mapping
        let mut rng_mapping =
            (0..addr_length_ as u64 * number_of_hashtables_ as u64).collect::<Vec<u64>>();
        rng_mapping.shuffle(&mut thread_rng());

        Wisard::<T> {
            discs: HashMap::new(),
            addr_length: addr_length_,
            number_of_hashtables: number_of_hashtables_,
            mapping: rng_mapping,
            last_rank: 0,
            rank_tables: HashMap::new(),
            bleach: bleach_,
            phantom: PhantomData,
        }
    }

    pub fn erase_and_change_hyperparameters(
        &mut self,
        number_of_hashtables: u16,
        addr_length: u16,
        bleach: u16,
    ) {
        self.erase();
        // randomizes the mapping
        let mut rng_mapping =
            (0..addr_length as u64 * number_of_hashtables as u64).collect::<Vec<u64>>();
        rng_mapping.shuffle(&mut thread_rng());

        self.discs = HashMap::new();
        self.addr_length = addr_length;
        self.number_of_hashtables = number_of_hashtables;
        self.mapping = rng_mapping;
        self.last_rank = 0;
        self.rank_tables = HashMap::new();
        self.bleach = bleach;
    }

    fn ranks(&mut self, samples: Vec<T>) -> Vec<u64>
    where
        T: PartialOrd + Copy,
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
                .map(|x| (x.0 as u64, x.1))
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

    pub fn train(&mut self, data: Vec<T>, label: String)
    where
        T: PartialOrd + Copy,
    {
        if !self.discs.contains_key(&label) {
            self.discs
                .insert(label.clone(), Discriminator::new(self.number_of_hashtables));
        }

        let samples = self.mapping.clone();
        let samples = samples
            .iter()
            .map(|&i| *data.get(i as usize).unwrap())
            .collect();
        let addresses: Vec<u64> = self.ranks(samples);
        let disc = self.discs.get_mut(&label).unwrap();
        disc.train(addresses);
    }

    pub fn classify(&mut self, data: Vec<T>) -> (String, f64, f64)
    where
        T: PartialOrd + Copy,
    {
        let samples = self.mapping.clone();
        let samples = samples
            .iter()
            .map(|&i| *data.get(i as usize).unwrap())
            .collect();
        let addresses: Vec<u64> = self.ranks(samples);
        let discs = &self.discs;
        let mut votes: Vec<(String, (u64, u64))> = discs
            .iter()
            .map(|d| (d.0.to_string(), d.1.classify(&addresses, self.bleach)))
            .collect();
        votes.sort_by(|a, b| (a.1).0.partial_cmp(&(b.1).0).unwrap());

        let biggest = votes.len().checked_sub(1).map(|i| &votes[i]).unwrap();
        let second_biggest = votes.len().checked_sub(2).map(|i| &votes[i]).unwrap();

        (
            biggest.0.clone(),                                        // elected label
            (biggest.1 .0 as f64 / self.number_of_hashtables as f64), // "acc"
            (biggest.1 .0 as f64 - second_biggest.1 .0 as f64) / biggest.1 .0 as f64, // confidence
        )
    }
    pub fn save(&self) -> Vec<u8> {
        let encoded: Vec<u8> = bincode::serialize(&self).unwrap();
        encoded
    }
    pub fn load(&mut self, stream: &[u8]) {
        let decoded: Wisard<T> = bincode::deserialize(stream).unwrap();
        self.discs = decoded.discs;
        self.addr_length = decoded.addr_length;
        self.number_of_hashtables = decoded.number_of_hashtables;
        self.mapping = decoded.mapping;
        self.last_rank = decoded.last_rank;
        self.rank_tables = decoded.rank_tables;
        self.bleach = decoded.bleach;
    }
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) {
        let mut file = File::create(path).unwrap();
        bincode::serialize_into(&mut file, &self).unwrap();
    }
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) {
        let file = File::open(path).unwrap();
        let decoded: Wisard<T> = bincode::deserialize_from(file).unwrap();
        self.discs = decoded.discs;
        self.addr_length = decoded.addr_length;
        self.number_of_hashtables = decoded.number_of_hashtables;
        self.mapping = decoded.mapping;
        self.last_rank = decoded.last_rank;
        self.rank_tables = decoded.rank_tables;
        self.bleach = decoded.bleach;
    }
    pub fn erase(&mut self) {
        self.mapping.shuffle(&mut thread_rng());
        self.discs = HashMap::new();
        self.last_rank = 0;
        self.rank_tables = HashMap::new()
    }
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_lib_ranks() {
        // this test verifies that ranks is able to push address to rank_tables
        let mut wis = Wisard::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks(samples);
        assert!(!wis.rank_tables.is_empty());
    }
    #[test]
    fn test_lib_rank_table_length() {
        // this test ensures that the same addresses aren't pushed into the rank_tables
        // repeatedly
        let mut wis = Wisard::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks(samples);
        let length1 = wis.rank_tables.len();
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        wis.ranks(samples);
        let length2 = wis.rank_tables.len();
        assert_eq!(length1, length2);
    }

    #[test]
    fn test_lib_rank_addresses() {
        // this test verifies that for each new piece of data, a correct rank is attributed
        let mut wis = Wisard::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let addresses = wis.ranks(samples);
        assert_eq!(addresses, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }
    #[test]
    fn test_lib_rank_different_addresses() {
        // this test verifies that small changes in data get close addresses
        let mut wis = Wisard::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let addresses = wis.ranks(samples);
        assert_eq!(addresses, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 205,
        ];
        let addresses = wis.ranks(samples);
        assert_eq!(addresses, vec![0, 1, 2, 3, 4, 5, 6, 7, 9]);
    }

    #[test]
    fn test_save_load() {
        use std::fs;
        fs::create_dir_all("profiling/").unwrap();

        let mut wis = Wisard::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let _ = wis.ranks(samples);

        wis.save_to_file("profiling/weigths_u8.bin");

        let mut decoded = Wisard::<u8>::new(28, 8, 0);
        decoded.load_from_file("profiling/weigths_u8.bin");
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 205,
        ];
        let decoded_addresses = decoded.ranks(samples);

        println!("{:?}", decoded_addresses);

        assert_eq!(vec![0, 1, 2, 3, 4, 5, 6, 7, 9], decoded_addresses);

        fs::remove_file("profiling/weigths_u8.bin").unwrap();
        fs::remove_dir_all("profiling/").unwrap();
    }

    #[test]
    fn test_erase() {
        let mut wis = Wisard::new(28, 8, 0);
        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 47,
        ];
        let _ = wis.ranks(samples);

        wis.erase();

        let samples = vec![
            52, 70, 64, 199, 7, 133, 5, 194, 16, 104, 41, 147, 42, 77, 188, 140, 148, 160, 6, 87,
            107, 73, 168, 95, 63, 11, 2, 49, 130, 43, 92, 110, 13, 157, 125, 6, 93, 119, 86, 85,
            103, 27, 124, 65, 9, 195, 21, 130, 192, 32, 136, 34, 70, 89, 84, 167, 175, 148, 116,
            177, 161, 134, 98, 30, 190, 205,
        ];
        let decoded_addresses = wis.ranks(samples);

        assert_eq!(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], decoded_addresses);
    }
}
