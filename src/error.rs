use reqwest;
use serde_json;
use std::error::Error;
use std::fmt;

pub type GymResult<T> = Result<T, GymError>;

#[derive(Debug)]
pub enum GymError {
	Connection(reqwest::Error),
	Json(serde_json::Error),
	InstanceId(String),
}

impl fmt::Display for GymError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			GymError::Connection(ref err) => fmt::Display::fmt(err, f),
			GymError::Json(ref err) => fmt::Display::fmt(err, f),
			GymError::InstanceId(ref message) => fmt::Display::fmt(message, f),
		}
	}
}

impl Error for GymError {
	fn description(&self) -> &str {
		match *self {
			GymError::Connection(ref err) => err.description(),
			GymError::Json(ref err) => err.description(),
			GymError::InstanceId(ref message) => message,
		}
	}
}

impl From<reqwest::Error> for GymError {
	fn from(err: reqwest::Error) -> GymError {
		GymError::Connection(err)
	}
}

impl From<serde_json::Error> for GymError {
	fn from(err: serde_json::Error) -> GymError {
		GymError::Json(err)
	}
}
