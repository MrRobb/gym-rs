use cpython::{GILGuard, ObjectProtocol, PyModule, Python};

use crate::environment::Environment;
use crate::space_template::SpaceTemplate;

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
		let gym = py.import("gym").expect("Error: import gym");
		let version = gym
			.get(py, "__version__")
			.expect("Unable to call gym.__version__")
			.extract(py)
			.expect("Unable to call gym.__version__");

		Self { gil, gym, version }
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
				&env.getattr(py, "observation_space")
					.expect("Unable to get attribute 'observation_space'"),
			),
			action_space: SpaceTemplate::extract_template(
				&env.getattr(py, "action_space")
					.expect("Unable to get attribute 'action_space'"),
			),
			env,
		}
	}

	pub fn version(&self) -> &str {
		self.version.as_str()
	}
}
