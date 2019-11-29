extern crate cpython;
extern crate ndarray;
extern crate num;
extern crate rand;

use cpython::*;
use rand::Rng;

type DiscreteType = usize;
type VectorType<T> = ndarray::Array1<T>;
pub type Action = SpaceData;
pub type Observation = SpaceData;
pub type Reward = f64;

pub struct State {
	pub observation: SpaceData,
	pub reward: f64,
	pub is_done: bool,
}

pub enum SpaceTemplate {
	DISCRETE {
		n: DiscreteType,
	},
	BOX {
		high: Vec<f64>,
		low: Vec<f64>,
		shape: Vec<usize>,
	},
	Tuple {
		shape: Vec<usize>,
	},
}

#[derive(Debug)]
pub enum SpaceData {
	DISCRETE(DiscreteType),
	BOX(VectorType<f64>),
	TUPLE(VectorType<SpaceData>),
}

pub struct Environment<'a> {
	gil: &'a GILGuard,
	env: PyObject,
	observation_space: SpaceTemplate,
	action_space: SpaceTemplate,
}

pub struct GymClient {
	gil: GILGuard,
	gym: PyModule,
}

impl SpaceData {
	pub fn get_discrete(self) -> Option<DiscreteType> {
		match self {
			SpaceData::DISCRETE(n) => Some(n),
			_ => None,
		}
	}

	pub fn get_box(self) -> Option<VectorType<f64>> {
		match self {
			SpaceData::BOX(v) => Some(v),
			_ => None,
		}
	}

	pub fn get_tuple(self) -> Option<VectorType<SpaceData>> {
		match self {
			SpaceData::TUPLE(s) => Some(s),
			_ => None,
		}
	}
}

impl SpaceTemplate {
	fn extract_data(&self, pyo: PyObject) -> SpaceData {
		let gil = Python::acquire_gil();
		let py = gil.python();

		match self {
			SpaceTemplate::DISCRETE { .. } => {
				let n = pyo
					.extract::<DiscreteType>(py)
					.expect("Unable to convert observation to u64");
				SpaceData::DISCRETE(n)
			},
			SpaceTemplate::BOX { .. } => {
				let v = pyo
					.extract::<Vec<f64>>(py)
					.expect("Unable to convert observation to Vec");
				SpaceData::BOX(v.into())
			},
			SpaceTemplate::Tuple { .. } => {
				unimplemented!();
			},
		}
	}

	pub fn sample(&self) -> SpaceData {
		let mut rng = rand::thread_rng();
		match self {
			SpaceTemplate::DISCRETE { n } => SpaceData::DISCRETE(rng.gen_range(0, n)),
			SpaceTemplate::BOX { high, low, shape } => {
				let dimensions = shape.len();
				let mut v = vec![];
				for d in 0..dimensions {
					for _ in 0..shape[d] {
						v.push(rng.gen_range(low[d], high[d]));
					}
				}
				SpaceData::BOX(v.into())
			},
			SpaceTemplate::Tuple { .. } => unimplemented!(),
		}
	}
}

impl FromPyObject<'_> for SpaceTemplate {
	fn extract(py: Python, obj: &PyObject) -> Result<Self, PyErr> {
		let class = obj.getattr(py, "__class__")?;
		let name = class.getattr(py, "__name__")?.extract::<String>(py)?;
		match name.as_ref() {
			"Discrete" => {
				let n = obj.getattr(py, "n")?.extract::<usize>(py)?;
				Ok(SpaceTemplate::DISCRETE { n })
			},
			"Box" => {
				let high = obj.getattr(py, "high")?.extract::<Vec<f64>>(py)?;
				let low = obj.getattr(py, "low")?.extract::<Vec<f64>>(py)?;
				let shape = obj.getattr(py, "shape")?.extract::<Vec<usize>>(py)?;
				debug_assert_eq!(high.len(), low.len());
				debug_assert_eq!(low.len(), shape.iter().product());
				high.iter().zip(low.iter()).for_each(|(h, l)| assert!(h > l));
				Ok(SpaceTemplate::BOX { high, low, shape })
			},
			"Tuple" => unimplemented!(),
			_ => unreachable!(),
		}
	}
}

impl<'a> Environment<'a> {
	pub fn reset(&self) -> SpaceData {
		let py = self.gil.python();
		let result = self.env.call_method(py, "reset", NoArgs, None).expect("Error: reset()");
		// println!("Reset: {:?}", result);
		self.observation_space.extract_data(result)
	}

	pub fn render(&self) {
		let py = self.gil.python();
		self.env
			.call_method(py, "render", NoArgs, None)
			.expect("Error: render()");
	}

	pub fn step(&self, action: &Action) -> State {
		let py = self.gil.python();
		let result = match action {
			Action::DISCRETE(n) => self.env.call_method(py, "step", (n,), None).expect("Error: step()"),
			Action::BOX(_) => unimplemented!(),
			Action::TUPLE(_) => unimplemented!(),
		};
		State {
			observation: self
				.observation_space
				.extract_data(result.get_item(py, 0).expect("Error: extract step result")),
			reward: result
				.get_item(py, 1)
				.expect("Error: extract step result")
				.extract(py)
				.unwrap(),
			is_done: result
				.get_item(py, 2)
				.expect("Error: extract step result")
				.extract(py)
				.unwrap(),
		}
	}

	pub fn close(&self) {
		let py = self.gil.python();
		let result = self.env.call_method(py, "close", NoArgs, None).expect("Error: close()");
		println!("Close: {:?}", result);
	}

	/// Returns the number of allowed actions for this environment.
	pub fn action_space(&self) -> &SpaceTemplate {
		&self.action_space
	}

	/// Returns the shape of the observation tensors.
	pub fn observation_space(&self) -> &SpaceTemplate {
		&self.observation_space
	}
}

impl Default for GymClient {
	fn default() -> Self {
		// Get python
		let gil = Python::acquire_gil();
		let py = gil.python();

		// Set argv[0] -> otherwise render() fails
		let sys = py.import("sys").expect("Error: import sys");
		sys.get(py, "argv")
			.expect("Error: sys.argv")
			.call_method(py, "append", ("",), None)
			.expect("Error: sys.argv.append('')");

		// Import gym
		let gym = py.import("gym").expect("Error: import gym");
		let version = gym.get(py, "__version__").expect("Error: gym.__version__");
		println!("gym version: {:?}", version);

		GymClient { gil, gym }
	}
}

impl GymClient {
	pub fn make(&self, env_id: &str, _seed: Option<u64>) -> Environment {
		let py = self.gil.python();
		let env = self.gym.call(py, "make", (env_id,), None).expect("Error: make()");
		Environment {
			gil: &self.gil,
			observation_space: env
				.getattr(py, "observation_space")
				.unwrap()
				.extract::<SpaceTemplate>(py)
				.unwrap(),
			action_space: env
				.getattr(py, "action_space")
				.unwrap()
				.extract::<SpaceTemplate>(py)
				.unwrap(),
			env,
		}
	}
}

#[cfg(test)]
mod tests {

	use super::*;

	const _ENVS: &[&str] = &[
		"KellyCoinflipGeneralized-v0",
		"KellyCoinflip-v0",
		"Blackjack-v0",
		"LunarLanderContinuous-v2",
		"Copy-v0",
		"Bowling-ram-v0",
		"VideoPinball-v0",
		"Reverse-v0",
		"ReversedAddition-v0",
		"ReversedAddition3-v0",
		"RepeatCopy-v0",
		"DuplicatedInput-v0",
	];

	#[test]
	fn test_1_new() {
		let _client = GymClient::default();
	}

	#[test]
	fn test_2_make() {
		let client = GymClient::default();
		client.make("CartPole-v0", None);
	}

	#[test]
	fn test_4_reset() {
		let client = GymClient::default();
		let env = client.make("CartPole-v0", None);
		env.reset();
	}

	#[test]
	fn test_5_step() {
		let client = GymClient::default();
		let env = client.make("CartPole-v0", None);
		env.reset();
	}
}
