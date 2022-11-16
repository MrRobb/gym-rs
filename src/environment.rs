use cpython::{GILGuard, NoArgs, ObjectProtocol, PyObject, PyTuple, ToPyObject};

use crate::error::GymError;
use crate::space_data::SpaceData;
use crate::space_template::SpaceTemplate;
use crate::{Action, State};

pub struct Environment<'a> {
	pub gil: &'a GILGuard,
	pub env: PyObject,
	pub observation_space: SpaceTemplate,
	pub action_space: SpaceTemplate,
}

impl<'a> Environment<'a> {
	pub fn reset(&self, seed: Option<u64>) -> Result<(SpaceData, PyObject), GymError> {
		let py = self.gil.python();
		let args = match seed {
			Some(seed) => (seed,).into_py_object(py),
			None => NoArgs.into_py_object(py),
		};
		let result = self
			.env
			.call_method(py, "reset", args, None)
			.expect("Unable to call 'reset'");
		let observation = self
			.observation_space
			.extract_data(&result.get_item(py, 0).map_err(|_| GymError::WrongResetResult)?)?;
		let info = result.get_item(py, 1).map_err(|_| GymError::WrongResetResult)?;
		Ok((observation, info))
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
			Action::Discrete(n) => self
				.env
				.call_method(py, "step", (n,), None)
				.map_err(|_| GymError::InvalidAction)?,
			Action::Box(v) => {
				let vv = v.to_vec();
				self.env
					.call_method(py, "step", (vv,), None)
					.map_err(|_| GymError::InvalidAction)?
			},
			Action::Tuple(spaces) => {
				let vpyo = spaces.to_vec().into_iter().map(SpaceData::into_pyo).collect::<Vec<_>>();
				let tuple_pyo = PyTuple::new(py, &vpyo);
				self.env
					.call_method(py, "step", (tuple_pyo,), None)
					.map_err(|_| GymError::InvalidAction)?
			},
		};

		let s = State {
			observation: self
				.observation_space
				.extract_data(&result.get_item(py, 0).map_err(|_| GymError::WrongStepResult)?)?,
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
			is_truncated: result
				.get_item(py, 3)
				.map_err(|_| GymError::WrongStepResult)?
				.extract(py)
				.map_err(|_| GymError::WrongStepResult)?,
		};

		Ok(s)
	}

	pub fn close(&self) {
		let py = self.gil.python();
		let _res = self
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
