extern crate gym;
use gym::GymClient;

fn main() {
	let client = GymClient::new("127.0.0.1".parse().unwrap(), 8000).unwrap();
	println!("Already running environments:\n{:?}\n", client.get_envs().unwrap());

	let env = client
		.make("CartPole-v0".into(), None)
		.expect("Could not make environment");

	println!("Observation space:\n{:?}\n", env.observation_space());
	println!("Action space:\n{:?}\n", env.action_space());

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
}
