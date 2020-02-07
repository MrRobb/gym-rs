#[macro_use]
extern crate failure;
extern crate cpython;
extern crate ndarray;
extern crate rand;

use cpython::*;
use failure::Fail;
use rand::Rng;

type DiscreteType = usize;
type VectorType<T> = ndarray::Array1<T>;
pub type Action = SpaceData;
pub type Observation = SpaceData;
pub type Reward = f64;

#[derive(Debug, Fail)]
pub enum GymError {
	#[fail(display = "Invalid action")]
	InvalidAction,
	#[fail(display = "Invalid conversion")]
	InvalidConversion,
	#[fail(display = "Wrong type")]
	WrongType,
	#[fail(display = "Unable to parse step result")]
	WrongStepResult,
}

pub struct State {
	pub observation: SpaceData,
	pub reward: f64,
	pub is_done: bool,
}

#[derive(Debug)]
pub enum SpaceTemplate {
	DISCRETE {
		n: DiscreteType,
	},
	BOX {
		high: Vec<f64>,
		low: Vec<f64>,
		shape: Vec<usize>,
	},
	TUPLE {
		spaces: Vec<SpaceTemplate>,
	},
}

#[derive(Debug, Clone)]
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
	version: String,
}

impl SpaceData {
	pub fn get_discrete(self) -> Result<DiscreteType, GymError> {
		match self {
			SpaceData::DISCRETE(n) => Ok(n),
			_ => Err(GymError::WrongType),
		}
	}

	pub fn get_box(self) -> Result<VectorType<f64>, GymError> {
		match self {
			SpaceData::BOX(v) => Ok(v),
			_ => Err(GymError::WrongType),
		}
	}

	pub fn get_tuple(self) -> Result<VectorType<SpaceData>, GymError> {
		match self {
			SpaceData::TUPLE(s) => Ok(s),
			_ => Err(GymError::WrongType),
		}
	}

	pub fn into_pyo(self) -> Result<PyObject, GymError> {
		let gil = Python::acquire_gil();
		let py = gil.python();
		Ok(match self {
			SpaceData::DISCRETE(n) => n.into_py_object(py).into_object(),
			SpaceData::BOX(v) => v.to_vec().into_py_object(py).into_object(),
			SpaceData::TUPLE(spaces) => {
				let vpyo = spaces
					.to_vec()
					.into_iter()
					.map(|s| s.into_pyo().expect("Unable to parse tuple"))
					.collect::<Vec<_>>();
				vpyo.into_py_object(py).into_object()
			},
		})
	}
}

impl SpaceTemplate {
	fn extract_data(&self, pyo: PyObject) -> Result<SpaceData, GymError> {
		let gil = Python::acquire_gil();
		let py = gil.python();

		match self {
			SpaceTemplate::DISCRETE { .. } => {
				let n = pyo
					.extract::<DiscreteType>(py)
					.map_err(|_| GymError::InvalidConversion)?;
				Ok(SpaceData::DISCRETE(n))
			},
			SpaceTemplate::BOX { .. } => {
				let v = pyo
					.call_method(py, "flatten", NoArgs, None)
					.map_err(|_| GymError::InvalidConversion)?
					.extract::<Vec<f64>>(py)
					.map_err(|_| GymError::InvalidConversion)?;
				Ok(SpaceData::BOX(v.into()))
			},
			SpaceTemplate::TUPLE { .. } => {
				let mut tuple = vec![];
				let mut i = 0;
				let mut item = pyo.get_item(py, i);
				while item.is_ok() {
					let pyo_item = self.extract_data(item.unwrap())?;
					tuple.push(pyo_item);
					i += 1;
					item = pyo.get_item(py, i);
				}
				Ok(SpaceData::TUPLE(tuple.into()))
			},
		}
	}

	fn extract_template(pyo: PyObject) -> SpaceTemplate {
		let gil = Python::acquire_gil();
		let py = gil.python();

		let class = pyo
			.getattr(py, "__class__")
			.expect("Unable to extract __class__ (this should never happen)");

		let name = class
			.getattr(py, "__name__")
			.expect("Unable to extract __name__ (this should never happen)")
			.extract::<String>(py)
			.expect("Unable to extract __name__ (this should never happen)");

		match name.as_ref() {
			"Discrete" => {
				let n = pyo
					.getattr(py, "n")
					.expect("Unable to get attribute 'n'")
					.extract::<usize>(py)
					.expect("Unable to convert 'n' to usize");
				SpaceTemplate::DISCRETE { n }
			},
			"Box" => {
				let high = pyo
					.getattr(py, "high")
					.expect("Unable to get attribute 'high'")
					.call_method(py, "flatten", NoArgs, None)
					.expect("Unable to call 'flatten'")
					.extract::<Vec<f64>>(py)
					.expect("Unable to convert 'high' to Vec<f64>");

				let low = pyo
					.getattr(py, "low")
					.expect("Unable to get attribute 'low'")
					.call_method(py, "flatten", NoArgs, None)
					.expect("Unable to call 'flatten'")
					.extract::<Vec<f64>>(py)
					.expect("Unable to convert 'low' to Vec<f64>");

				let shape = pyo
					.getattr(py, "shape")
					.expect("Unable to get attribute 'shape'")
					.extract::<Vec<usize>>(py)
					.expect("Unable to convert 'shape' to Vec<f64>");

				debug_assert_eq!(high.len(), low.len());
				debug_assert_eq!(low.len(), shape.iter().product());
				high.iter().zip(low.iter()).for_each(|(h, l)| debug_assert!(h > l));

				SpaceTemplate::BOX { high, low, shape }
			},
			"Tuple" => {
				let mut i = 0;
				let mut tuple = vec![];
				let mut item = pyo.get_item(py, i);

				while item.is_ok() {
					let pyo_item = item.unwrap();
					let space = SpaceTemplate::extract_template(pyo_item);
					tuple.push(space);
					i += 1;
					item = pyo.get_item(py, i);
				}

				SpaceTemplate::TUPLE { spaces: tuple }
			},
			_ => unreachable!(),
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
			SpaceTemplate::TUPLE { spaces } => {
				let mut tuple = vec![];
				for space in spaces {
					let sample = space.sample();
					tuple.push(sample);
				}
				SpaceData::TUPLE(tuple.into())
			},
		}
	}
}

impl<'a> Environment<'a> {

	pub fn seed(&self, seed: u64) {
		let py = self.gil.python();
		self.env.call_method(py, "seed", (seed,), None).expect("Unable to call 'seed'");
	}

	pub fn reset(&self) -> Result<SpaceData, GymError> {
		let py = self.gil.python();
		let result = self
			.env
			.call_method(py, "reset", NoArgs, None)
			.expect("Unable to call 'reset'");
		self.observation_space.extract_data(result)
	}

	pub fn render(&self) {
		let py = self.gil.python();
		self.env
			.call_method(py, "render", NoArgs, None)
			.expect("Unable to call 'render'");
	}

	pub fn step(&self, action: &Action) -> Result<State, GymError> {
		let py = self.gil.python();
		let result = match action {
			Action::DISCRETE(n) => self
				.env
				.call_method(py, "step", (n,), None)
				.map_err(|_| GymError::InvalidAction)?,
			Action::BOX(v) => {
				let vv = v.to_vec();
				self.env
					.call_method(py, "step", (vv,), None)
					.map_err(|_| GymError::InvalidAction)?
			},
			Action::TUPLE(spaces) => {
				let vpyo = spaces
					.to_vec()
					.into_iter()
					.map(|s| s.into_pyo().unwrap())
					.collect::<Vec<_>>();
				let tpyo = PyTuple::new(py, &vpyo);
				self.env
					.call_method(py, "step", (tpyo,), None)
					.map_err(|_| GymError::InvalidAction)?
			},
		};

		let s = State {
			observation: self
				.observation_space
				.extract_data(result.get_item(py, 0).map_err(|_| GymError::WrongStepResult)?)?,
			reward: result
				.get_item(py, 1)
				.map_err(|_| GymError::WrongStepResult)?
				.extract(py)
				.map_err(|_| GymError::WrongStepResult)?,
			is_done: result
				.get_item(py, 2)
				.map_err(|_| GymError::WrongStepResult)?
				.extract(py)
				.map_err(|_| GymError::WrongStepResult)?,
		};

		Ok(s)
	}

	pub fn close(&self) {
		let py = self.gil.python();
		let _ = self
			.env
			.call_method(py, "close", NoArgs, None)
			.expect("Unable to call 'close'");
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

		match sys.get(py, "argv") {
			Result::Ok(argv) => {
				argv.call_method(py, "append", ("",), None).expect("Error: sys.argv.append('')");
			},
			Result::Err(_) => {},
		};

		// Import gym
		let gym = py.import("gym").expect("Error: import gym");
		let version = gym
			.get(py, "__version__")
			.expect("Unable to call gym.__version__")
			.extract(py)
			.expect("Unable to call gym.__version__");

		GymClient { gil, gym, version }
	}
}

impl GymClient {
	pub fn make(&self, env_id: &str) -> Environment {
		let py = self.gil.python();
		let env = self
			.gym
			.call(py, "make", (env_id,), None)
			.expect("Unable to call 'make'");

		Environment {
			gil: &self.gil,
			observation_space: SpaceTemplate::extract_template(
				env.getattr(py, "observation_space")
					.expect("Unable to get attribute 'observation_space'"),
			),
			action_space: SpaceTemplate::extract_template(
				env.getattr(py, "action_space")
					.expect("Unable to get attribute 'action_space'"),
			),
			env,
		}
	}

	pub fn version(&self) -> &str {
		self.version.as_str()
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
	fn test_gym_client() {
		let _client = GymClient::default();
	}

	#[test]
	fn test_make() {
		let client = GymClient::default();
		client.make("CartPole-v1");
	}

	#[test]
	fn test_seed() {
		let client = GymClient::default();
		let env = client.make("FrozenLake-v0");
		env.seed(1002);
		let obs = env.reset().unwrap();
		assert_eq!(0, obs.get_discrete().unwrap());
		let action = SpaceData::DISCRETE(1);
		let state = env.step(&action).unwrap();
		assert_eq!(4, state.observation.get_discrete().unwrap());
	}

	#[test]
	fn test_reset() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
	}

	#[test]
	fn test_box_observation_3d() {
		let client = GymClient::default();
		let env = client.make("VideoPinball-v0");
		env.reset().unwrap();
		env.step(&env.action_space().sample()).unwrap();
	}

	#[test]
	fn test_step() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}

	#[test]
	#[should_panic]
	fn test_invalid_action() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
		let action = Action::DISCRETE(500); // invalid action
		env.step(&action).unwrap();
	}

	#[test]
	#[should_panic]
	fn test_wrong_type() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
		let _ = env.action_space().sample().get_box().unwrap();
	}

	#[test]
	fn test_box_action() {
		let client = GymClient::default();
		let env = client.make("BipedalWalker-v2");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}

	#[test]
	fn test_tuple_template() {
		let client = GymClient::default();
		let _ = client.make("Blackjack-v0");
	}

	#[test]
	fn test_tuple_obs() {
		let client = GymClient::default();
		let env = client.make("Blackjack-v0");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}

	#[test]
	fn test_tuple_action() {
		let client = GymClient::default();
		let env = client.make("RepeatCopy-v0");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}
}
