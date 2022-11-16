use gym::client::MakeOptions;

extern crate gym;

fn main() {
	let gym = gym::client::GymClient::default();
	let env = gym
		.make(
			"CartPole-v1",
			Some(MakeOptions {
				render_mode: Some(gym::client::RenderMode::Human),
				..Default::default()
			}),
		)
		.expect("Unable to create environment");

	for _ in 0..10 {
		env.reset(None).expect("Unable to reset");

		for _ in 0..100 {
			let action = env.action_space().sample();
			let state = env.step(&action).unwrap();
			env.render();
			if state.is_done {
				break;
			}
		}
	}

	env.close();
}
