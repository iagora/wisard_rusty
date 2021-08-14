use std::{error::Error, fmt};

#[derive(Debug)]
pub enum WisardError {
    WisardOutOfBounds,
    WisardValidationFailed(String),
    WisardIOError,
}

impl Error for WisardError {}

impl fmt::Display for WisardError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WisardError::WisardOutOfBounds => write!(f, "Aaaand we're out of bounds!"),
            WisardError::WisardValidationFailed(s) => {
                write!(f, "This file is not what I expected! {}", s)
            }
            WisardError::WisardIOError => write!(f, "IO Error!"),
        }
    }
}
