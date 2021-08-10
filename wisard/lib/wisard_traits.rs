use crate::errors::WisardError;
use std::path::Path;

pub trait WisardNetwork<T> {
    // fn new() -> Self
    // where
    //     T: PartialOrd + Copy + Send + Sync;
    fn train(&mut self, data: Vec<T>, label: String) -> Result<(), WisardError>
    where
        T: PartialOrd + Copy + Send + Sync;
    fn classify(&self, data: Vec<T>) -> Result<String, WisardError>
    where
        T: PartialOrd + Copy + Send + Sync;
    fn save(&self) -> Result<Vec<u8>, WisardError>;
    fn load(&mut self, stream: &[u8]) -> Result<(), WisardError>;
    fn save_to_file(&self, path: &Path) -> Result<(), WisardError>;
    fn load_from_file(&mut self, path: &Path) -> Result<(), WisardError>;
    fn erase(&mut self);
}
