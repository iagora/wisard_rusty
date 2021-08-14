use crate::error::WisardError;

pub trait WisardNetwork<T> {
    // fn new() -> Self
    // where
    //     T: PartialOrd + Copy + Send + Sync;
    fn get_info(&self) -> (u16, u16, u16, (u32, u32), Vec<u64>);
    fn change_hyperparameters(
        &mut self,
        number_of_hashtables: u16,
        addr_length: u16,
        bleach: u16,
        target_size: Option<(u32, u32)>,
        mapping: Option<Vec<u64>>,
    );
    fn target_size(&self) -> (u32, u32);
    fn train(&mut self, data: Vec<T>, label: String) -> Result<(), WisardError>
    where
        T: PartialOrd + Copy + Send + Sync;
    fn classify(&self, data: Vec<T>) -> Result<String, WisardError>
    where
        T: PartialOrd + Copy + Send + Sync;
    fn save(&self) -> Result<Vec<u8>, WisardError>;
    fn load(&mut self, stream: &[u8]) -> Result<(), WisardError>;
    fn erase(&mut self);
}
