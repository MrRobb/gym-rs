use cpython::{PyErr, PyObject};
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
	#[error("Invalid seed")]
	InvalidSeed,
	#[error("Invalid render mode")]
	InvalidRenderMode,
	#[error("Unable to make environment '{0}' with dict '{1:?}' (Error: {2:?})")]
	InvalidMake(String, Vec<(PyObject, PyObject)>, PyErr),
}
