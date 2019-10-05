
extern crate gym;
use gym::GymClient;

fn main() {
	let client = GymClient::new_quiet("http://localhost:5000".to_string()).unwrap();
	println!("Already running environments:\n{:?}\n", client.get_envs().unwrap());

	let env = match client.make("CartPole-v0") {
		Ok(env) => env,
		Err(msg) => panic!("Could not make environment because of error:\n{}", msg)
	};

	println!("Observation space:\n{:?}\n", env.observation_space());
	println!("Action space:\n{:?}\n", env.action_space());

	let _ = env.monitor_start("/tmp/random-agent-results".to_string(), true, false);

	for ep in 0..10 {
		let mut tot_reward = 0.;
		let _ = env.reset();
		loop {
			let action = env.action_space().sample();
			let state = env.step(action, true).unwrap();
			assert_eq!(state.observation.len(), env.observation_space().sample().len());
			tot_reward += state.reward;

			if state.done {
				break;
			}
		}
		println!("Finished episode {} with total reward {}", ep, tot_reward);
	}

	let _ = env.monitor_stop();
}