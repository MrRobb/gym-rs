use thiserror::Error;

#[derive(Debug, Error)]
pub enum GymError {
	#[error("Invalid action")]
	InvalidAction,
	#[error("Invalid conversion")]
	InvalidConversion,
	#[error("Wrong type")]
	WrongType,
	#[error("Unable to parse step result")]
	WrongStepResult,
	#[error("Unable to parse reset result")]
	WrongResetResult,
}
