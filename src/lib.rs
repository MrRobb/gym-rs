extern crate rand;
extern crate reqwest;
extern crate serde;
extern crate serde_json;

mod error;
mod space;

use serde::{Deserialize, Serialize};
use serde_json::value::from_value;
use serde_json::Value;
use std::collections::BTreeMap;

pub use self::error::{GymError, GymResult};
pub use self::space::Space;
use serde_json::json;
use std::net::IpAddr;
use std::process::{Child, Command, Stdio};

pub type Observation = Vec<f64>;
pub type Action = Vec<f64>;
pub type Reward = f64;

#[derive(Debug)]
pub struct State {
	pub observation: Observation,
	pub reward: Reward,
	pub done: bool,
	pub info: Value,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EnvReq {
	pub env_id: String,
	pub seed: Option<u64>,
}

pub struct Environment<'a> {
	client: &'a GymClient,
	instance_id: String,
	act_space: Space,
	obs_space: Space,
}

impl<'a> Environment<'a> {
	fn box_to_obs(bx: Vec<Value>, dimensions: usize) -> Observation {
		match dimensions {
			1 => bx
				.into_iter()
				.map(|x| x.as_f64().expect("Box cannot be converted into Vec<f64>"))
				.collect(),
			2 => bx
				.into_iter()
				.map(|x| -> Vec<f64> { from_value(x).unwrap() })
				.flatten()
				.collect(),
			3 => bx
				.into_iter()
				.map(|x| -> Vec<Vec<f64>> { from_value(x).unwrap() })
				.flatten()
				.flatten()
				.collect(),
			_ => bx
				.into_iter()
				.map(|x| Environment::box_to_obs(x.as_array().unwrap().to_owned(), dimensions - 1))
				.flatten()
				.collect(),
		}
	}

	fn tuple_to_obs(t: Vec<Value>) -> Observation {
		t.into_iter()
			.map(|x| match x {
				Value::Bool(b) => {
					if b {
						vec![1.0]
					}
					else {
						vec![0.0]
					}
				},
				Value::Number(n) => vec![n.as_f64().unwrap()],
				Value::Array(v) => Environment::tuple_to_obs(v),
				Value::Null => panic!("Tuple contained a null value"),
				Value::Object(_) => panic!("Tuple contained an object"),
				Value::String(_) => panic!("Tuple contained a string"),
			})
			.flatten()
			.collect()
	}

	fn value_to_obs(&self, s: Value) -> GymResult<Observation> {
		match self.observation_space() {
			Space::DISCRETE { .. } => match s["observation"].as_f64() {
				Some(value) => Ok(vec![value]),
				None => panic!("Discrete is not a float"),
			},
			Space::BOX { ref shape, .. } => match s["observation"].as_array() {
				Some(value) => Ok(Environment::box_to_obs(value.to_owned(), shape.len())),
				None => panic!("Box is not an array"),
			},
			Space::TUPLE { .. } => match s["observation"].as_array() {
				Some(value) => Ok(Environment::tuple_to_obs(value.to_owned())),
				None => panic!("Tuple is not an array"),
			},
		}
	}

	pub fn action_space(&self) -> &Space {
		&self.act_space
	}

	pub fn observation_space(&self) -> &Space {
		&self.obs_space
	}

	pub fn reset(&self) -> GymResult<Observation> {
		let path = "/v1/envs/".to_string() + &self.instance_id + "/reset/";
		let observation = self.client.post(path, &Value::Null)?;
		self.value_to_obs(observation)
	}

	pub fn step(&self, action: Action, render: bool) -> GymResult<State> {
		let request = match self.act_space {
			Space::DISCRETE { .. } => {
				debug_assert_eq!(action.len(), 1);
				json!({
					"render": render,
					"action": (action[0] as u64)
				})
			},
			Space::BOX { ref shape, .. } => {
				debug_assert_eq!(action.len(), shape.iter().map(|&x| x as usize).product::<usize>());
				json!({
					"render": render,
					"action": action
				})
			},
			Space::TUPLE { .. } => json!({
				"render": render,
				"action": action
			}),
		};

		let path = format!("/v1/envs/{}/step/", &self.instance_id);

		let state = self.client.post(path, &request)?;

		Ok(State {
			reward: state["reward"].as_f64().unwrap(),
			done: state["done"].as_bool().unwrap(),
			info: state["info"].clone(),
			observation: self.value_to_obs(state).unwrap(),
		})
	}

	pub fn monitor_start(&self, directory: String, force: bool, resume: bool) -> GymResult<Value> {
		let mut req = BTreeMap::new();
		req.insert("directory", Value::String(directory));
		req.insert("force", Value::Bool(force));
		req.insert("resume", Value::Bool(resume));

		let path = "/v1/envs/".to_string() + &self.instance_id + "/monitor/start/";
		self.client.post(path, &req)
	}

	pub fn monitor_stop(&self) -> GymResult<Value> {
		let path = "/v1/envs/".to_string() + &self.instance_id + "/monitor/close/";
		self.client.post(path, &Value::Null)
	}

	pub fn upload(&self, training_dir: String, api_key: String, algorithm_id: String) -> GymResult<Value> {
		let mut req = BTreeMap::new();
		req.insert("training_dir", training_dir);
		req.insert("api_key", api_key);
		req.insert("algorithm_id", algorithm_id);

		self.client.post("/v1/upload/".to_string(), &req)
	}
}

pub struct GymClient {
	address: String,
	handle: reqwest::Client,
	server: Child,
}

impl GymClient {
	pub fn new(addr: IpAddr, port: u16) -> GymResult<GymClient> {
		let ip_str = addr.to_string();
		let port_str = port.to_string();
		let child = std::thread::spawn(|| -> std::process::Child {
			Command::new("python3")
				.arg("server.py")
				.arg("--listen")
				.arg(ip_str)
				.arg("--port")
				.arg(port_str)
				.spawn()
				.expect("Could not initiate server")
		});

		std::thread::sleep(std::time::Duration::from_millis(500));

		Ok(GymClient {
			address: format!("http://{}:{}", addr.to_string(), &port.to_string()),
			handle: reqwest::Client::new(),
			server: child.join().unwrap(),
		})
	}

	pub fn new_quiet(addr: IpAddr, port: u16) -> GymResult<GymClient> {
		let ip_str = addr.to_string();
		let port_str = port.to_string();
		let child = std::thread::spawn(|| -> std::process::Child {
			Command::new("python3")
				.arg("server.py")
				.arg("--listen")
				.arg(ip_str)
				.arg("--port")
				.arg(port_str)
				.stdout(Stdio::null())
				.stderr(Stdio::null())
				.spawn()
				.expect("Could not initiate server")
		});

		std::thread::sleep(std::time::Duration::from_millis(1000));

		Ok(GymClient {
			address: format!("http://{}:{}", addr.to_string(), &port.to_string()),
			handle: reqwest::Client::new(),
			server: child.join().unwrap(),
		})
	}

	pub fn make(&self, env_id: String, seed: Option<u64>) -> GymResult<Environment> {
		let env = EnvReq { env_id, seed };

		let instance_id: String = match self.post("/v1/envs/".to_string(), &env) {
			Ok(response) => match response["instance_id"].as_str() {
				Some(id) => id.into(),
				None => return Err(GymError::InstanceId(response["message"].as_str().unwrap().into())),
			},
			Err(e) => return Err(GymError::InstanceId(e.to_string())),
		};

		let obs_space = self.get(format!("/v1/envs/{}/observation_space/", instance_id))?;
		let act_space = self.get(format!("/v1/envs/{}/action_space/", instance_id))?;

		Ok(Environment {
			client: self,
			instance_id: instance_id.to_string(),
			act_space: Space::from_json(&act_space["info"])?,
			obs_space: Space::from_json(&obs_space["info"])?,
		})
	}

	pub fn get_envs(&self) -> GymResult<BTreeMap<String, String>> {
		let json = self.get("/v1/envs/".to_string())?;
		Ok(from_value(json["all_envs"].clone()).unwrap())
	}

	fn post<T: Serialize>(&self, route: String, request: &T) -> GymResult<Value> {
		let url = self.address.clone() + &route;
		match self.handle.post(&url).json(request).send() {
			Ok(mut response) => match response.json() {
				Ok(js) => Ok(js),
				Err(e) => {
					println!("{:?}", response);
					Err(e.into())
				},
			},
			Err(e) => Err(e.into()),
		}
	}

	fn get(&self, route: String) -> GymResult<Value> {
		let url = self.address.clone() + &route;
		match self.handle.get(&url).send()?.json() {
			Ok(val) => Ok(val),
			Err(e) => Err(e.into()),
		}
	}
}

impl Drop for GymClient {
	fn drop(&mut self) {
		match self.server.kill() {
			Ok(_) => (),
			Err(e) => println!("Error killing Python process (server.py): {}", e),
		}
	}
}

#[cfg(test)]
mod tests {

	use super::*;

	const ENVS: &[&str] = &[
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
		let client = GymClient::new("127.0.0.1".parse().unwrap(), 8000);
		assert!(client.is_ok());
		assert!(client.unwrap().server.try_wait().unwrap().is_none());
	}

	#[test]
	fn test_2_new_quiet() {
		let client = GymClient::new_quiet("127.0.0.1".parse().unwrap(), 8001);
		assert!(client.is_ok());
		assert!(client.unwrap().server.try_wait().unwrap().is_none());
	}

	#[test]
	fn test_3_make() {
		let client = GymClient::new_quiet("127.0.0.1".parse().unwrap(), 8002).unwrap();
		let mut envs = vec![];

		for &env_id in ENVS.iter() {
			envs.push(
				client
					.make(env_id.into(), None)
					.unwrap_or_else(|e| panic!("Unable to create the environment: {}", e)),
			);
		}

		assert_eq!(envs.len(), client.get_envs().unwrap().len());
	}

	#[test]
	fn test_4_reset() {
		let client = GymClient::new_quiet("127.0.0.1".parse().unwrap(), 8003).unwrap();

		for &env_id in ENVS.iter() {
			let env = client.make(env_id.into(), None).unwrap();
			assert!(env.reset().is_ok());
		}
	}

	#[test]
	fn test_5_step() {
		let client = GymClient::new_quiet("127.0.0.1".parse().unwrap(), 8004).unwrap();

		for &env_id in ENVS.iter() {
			let env = client.make(env_id.into(), None).unwrap();
			let _ = env.reset().unwrap();
			let action = env.action_space().sample();
			println!("{} -> {:?}", env_id, action);
			let state = env.step(action, false).unwrap();
			assert_eq!(state.observation.len(), env.observation_space().sample().len());
		}
	}
}
