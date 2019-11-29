extern crate gym;

use gym::Action;

fn main() {
	let gym = gym::GymClient::default();
	let env = gym.make("CartPole-v1", None);

	for _ in 0..10 {
		env.reset();

		for _ in 0..100 {
			let action = env.action_space().sample().get_discrete().unwrap();
			let state = env.step(&Action::DISCRETE(action));
			env.render();
			if state.is_done {
				break;
			}
		}
	}
}
