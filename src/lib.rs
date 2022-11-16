#![allow(
	clippy::missing_errors_doc,
	clippy::missing_const_for_fn,
	clippy::missing_panics_doc,
	clippy::must_use_candidate,
	clippy::module_name_repetitions
)]

extern crate cpython;
extern crate ndarray;
extern crate rand;
use space_data::SpaceData;

pub mod client;
mod environment;
pub mod error;
pub mod space_data;
pub mod space_template;

type DiscreteType = usize;
type VectorType<T> = ndarray::Array1<T>;
pub type Action = SpaceData;
pub type Observation = SpaceData;
pub type Reward = f64;

pub struct State {
	pub observation: SpaceData,
	pub reward: f64,
	pub is_done: bool,
	pub is_truncated: bool,
}
