use cpython::{GILGuard, ObjectProtocol, PyDict, PyModule, Python};

use crate::space_template::SpaceTemplate;
use crate::{environment::Environment, error::GymError};

pub struct GymClient {
	pub gil: GILGuard,
	pub gym: PyModule,
	pub version: String,
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
				argv.call_method(py, "append", ("",), None)
					.expect("Error: sys.argv.append('')");
			},
			Result::Err(_) => {},
		};

		// Import gym
		let gym = py.import("gymnasium").expect("Error: import gym");
		let version = gym
			.get(py, "__version__")
			.expect("Unable to call gym.__version__")
			.extract(py)
			.expect("Unable to call gym.__version__");

		Self { gil, gym, version }
	}
}

impl GymClient {
	pub fn make(&self, env_id: &str, render_mode: Option<&str>) -> Result<Environment, GymError> {
		let py = self.gil.python();
		let dict = PyDict::new(py);
		if let Some(render_mode) = render_mode {
			dict.set_item(py, "render_mode", render_mode)
				.map_err(|_| GymError::InvalidRenderMode)?;
		}
		let env = self
			.gym
			.call(py, "make", (env_id,), Some(&dict))
			.map_err(|_| GymError::InvalidMake(env_id.to_owned(), dict.items(py)))?;

		Ok(Environment {
			gil: &self.gil,
			observation_space: SpaceTemplate::extract_template(
				&env.getattr(py, "observation_space")
					.expect("Unable to get attribute 'observation_space'"),
			),
			action_space: SpaceTemplate::extract_template(
				&env.getattr(py, "action_space")
					.expect("Unable to get attribute 'action_space'"),
			),
			env,
		})
	}

	pub fn version(&self) -> &str {
		self.version.as_str()
	}
}
