use cpython::{PyObject, Python, PythonObject, ToPyObject};

use crate::error::GymError;
use crate::{DiscreteType, VectorType};

#[derive(Debug, Clone)]
pub enum SpaceData {
	Discrete(DiscreteType),
	Box(VectorType<f64>),
	Tuple(VectorType<SpaceData>),
}

impl SpaceData {
	pub fn get_discrete(self) -> Result<DiscreteType, GymError> {
		match self {
			SpaceData::Discrete(n) => Ok(n),
			_ => Err(GymError::WrongType),
		}
	}

	pub fn get_box(self) -> Result<VectorType<f64>, GymError> {
		match self {
			SpaceData::Box(v) => Ok(v),
			_ => Err(GymError::WrongType),
		}
	}

	pub fn get_tuple(self) -> Result<VectorType<Self>, GymError> {
		match self {
			SpaceData::Tuple(s) => Ok(s),
			_ => Err(GymError::WrongType),
		}
	}

	pub fn into_pyo(self) -> PyObject {
		let gil = Python::acquire_gil();
		let py = gil.python();
		match self {
			SpaceData::Discrete(n) => n.into_py_object(py).into_object(),
			SpaceData::Box(v) => v.to_vec().into_py_object(py).into_object(),
			SpaceData::Tuple(spaces) => {
				let vpyo = spaces.to_vec().into_iter().map(Self::into_pyo).collect::<Vec<_>>();
				vpyo.into_py_object(py).into_object()
			},
		}
	}
}
