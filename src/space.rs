use super::error::GymResult;
use rand::{thread_rng, Rng};
use serde_json::value::from_value;
use serde_json::Value;
use std::iter;

fn tuples(sizes: &[u64]) -> Vec<Vec<u64>> {
	match sizes.len() {
		0 => vec![],
		1 => (0..sizes[0]).map(|x| vec![x]).collect(),
		_ => {
			let (&head, tail) = sizes.split_first().unwrap();
			(0..head)
				.flat_map(|x| {
					iter::repeat(x).zip(tuples(tail)).map(|(h, mut t)| {
						t.insert(0, h);
						t
					})
				})
				.collect()
		},
	}
}

#[derive(Debug, Clone)]
pub enum Space {
	DISCRETE {
		n: u64,
	},
	BOX {
		shape: Vec<u64>,
		high: Vec<f64>,
		low: Vec<f64>,
	},
	TUPLE {
		spaces: Vec<Space>,
	},
}

impl Space {
	pub(crate) fn from_json(info: &Value) -> GymResult<Space> {
		match info["name"].as_str().unwrap() {
			"Discrete" => {
				let n = info["n"].as_u64().unwrap();
				Ok(Space::DISCRETE { n })
			},
			"Box" => {
				let shape = from_value(info["shape"].clone())?;
				let high = from_value(info["high"].clone())?;
				let low = from_value(info["low"].clone())?;
				Ok(Space::BOX { shape, high, low })
			},
			"Tuple" => {
				let mut spaces = Vec::new();
				for space in info["spaces"].as_array().unwrap() {
					spaces.push(Space::from_json(space).unwrap());
				}
				Ok(Space::TUPLE { spaces })
			},
			e => panic!("Unrecognized space name: {}", e),
		}
	}

	pub fn sample(&self) -> Vec<f64> {
		let mut rng = thread_rng();
		match *self {
			Space::DISCRETE { n } => vec![(rng.gen::<u64>() % n) as f64],
			Space::BOX {
				ref shape,
				ref high,
				ref low,
			} => {
				let mut ret = Vec::with_capacity(shape.iter().map(|x| *x as usize).product());
				for (index, _) in tuples(shape).iter().enumerate() {
					ret.push(rng.gen_range(low[index], high[index]));
				}
				ret
			},
			Space::TUPLE { ref spaces } => {
				let mut ret = Vec::new();
				for space in spaces {
					ret.extend(space.sample());
				}
				ret
			},
		}
	}
}
