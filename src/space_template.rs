use cpython::{NoArgs, ObjectProtocol, PyObject, Python};
use rand::Rng;

use crate::error::GymError;
use crate::space_data::SpaceData;
use crate::DiscreteType;

#[derive(Debug)]
pub enum SpaceTemplate {
	Discrete {
		n: DiscreteType,
	},
	Box {
		high: Vec<f64>,
		low: Vec<f64>,
		shape: Vec<usize>,
	},
	Tuple {
		spaces: Vec<SpaceTemplate>,
	},
}

impl SpaceTemplate {
	pub fn extract_data(&self, pyo: &PyObject) -> Result<SpaceData, GymError> {
		let gil = Python::acquire_gil();
		let py = gil.python();

		match self {
			SpaceTemplate::Discrete { .. } => {
				let n = pyo
					.extract::<DiscreteType>(py)
					.map_err(|_| GymError::InvalidConversion)?;
				Ok(SpaceData::Discrete(n))
			},
			SpaceTemplate::Box { .. } => {
				let v = pyo
					.call_method(py, "flatten", NoArgs, None)
					.map_err(|_| GymError::InvalidConversion)?
					.extract::<Vec<f64>>(py)
					.map_err(|_| GymError::InvalidConversion)?;
				Ok(SpaceData::Box(v.into()))
			},
			SpaceTemplate::Tuple { .. } => {
				let mut tuple = vec![];
				let mut i = 0;
				let mut item = pyo.get_item(py, i);
				while item.is_ok() {
					let pyo_item = self.extract_data(&item.unwrap())?;
					tuple.push(pyo_item);
					i += 1;
					item = pyo.get_item(py, i);
				}
				Ok(SpaceData::Tuple(tuple.into()))
			},
		}
	}

	pub fn extract_template(pyo: &PyObject) -> Self {
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
				Self::Discrete { n }
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

				Self::Box { high, low, shape }
			},
			"Tuple" => {
				let mut i = 0;
				let mut tuple = vec![];
				let mut item = pyo.get_item(py, i);

				while item.is_ok() {
					let pyo_item = item.unwrap();
					let space = Self::extract_template(&pyo_item);
					tuple.push(space);
					i += 1;
					item = pyo.get_item(py, i);
				}

				Self::Tuple { spaces: tuple }
			},
			_ => unreachable!(),
		}
	}

	pub fn sample(&self) -> SpaceData {
		let mut rng = rand::thread_rng();
		match self {
			SpaceTemplate::Discrete { n } => SpaceData::Discrete(rng.gen_range(0..*n)),
			SpaceTemplate::Box { high, low, shape } => {
				let dimensions = shape.len();
				let mut v = vec![];
				for d in 0..dimensions {
					for _ in 0..shape[d] {
						v.push(rng.gen_range(low[d]..high[d]));
					}
				}
				SpaceData::Box(v.into())
			},
			SpaceTemplate::Tuple { spaces } => {
				let mut tuple = vec![];
				for space in spaces {
					let sample = space.sample();
					tuple.push(sample);
				}
				SpaceData::Tuple(tuple.into())
			},
		}
	}
}
