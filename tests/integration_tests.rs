#[cfg(test)]
mod tests {
	use gym::client::GymClient;
	use gym::space_data::SpaceData;
	use gym::Action;

	#[test]
	fn test_gym_client() {
		let _client = GymClient::default();
	}

	#[test]
	fn test_make() {
		let client = GymClient::default();
		client.make("CartPole-v1");
	}

	#[test]
	fn test_seed() {
		let client = GymClient::default();
		let env = client.make("FrozenLake-v1");
		env.seed(1002);
		let (obs, _) = env.reset().unwrap();
		assert_eq!(0, obs.get_discrete().unwrap());
		let action = SpaceData::Discrete(1);
		let state = env.step(&action).unwrap();
		assert_eq!(4, state.observation.get_discrete().unwrap());
	}

	#[test]
	fn test_reset() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
	}

	#[test]
	fn test_box_observation_3d() {
		let client = GymClient::default();
		let env = client.make("VideoPinball-v4");
		env.reset().unwrap();
		env.step(&env.action_space().sample()).unwrap();
	}

	#[test]
	fn test_step() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}

	#[test]
	#[should_panic]
	fn test_invalid_action() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
		let action = Action::Discrete(500); // invalid action
		env.step(&action).unwrap();
	}

	#[test]
	#[should_panic]
	fn test_wrong_type() {
		let client = GymClient::default();
		let env = client.make("CartPole-v1");
		env.reset().unwrap();
		let _res = env.action_space().sample().get_box().unwrap();
	}

	#[test]
	fn test_box_action() {
		let client = GymClient::default();
		let env = client.make("BipedalWalker-v3");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}

	#[test]
	fn test_tuple_template() {
		let client = GymClient::default();
		let _res = client.make("Blackjack-v1");
	}

	#[test]
	fn test_tuple_obs() {
		let client = GymClient::default();
		let env = client.make("Blackjack-v1");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}

	#[test]
	fn test_tuple_action() {
		let client = GymClient::default();
		let env = client.make("RepeatCopy-v0");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
	}

	#[test]
	fn test_gym_version() {
		let client = GymClient::default();
		assert!(!client.version().is_empty());
	}

	#[test]
	fn test_render() {
		let client = GymClient::default();
		let env = client.make("FrozenLake-v1");
		env.reset().unwrap();
		let action = env.action_space().sample();
		env.step(&action).unwrap();
		env.render();
	}

	#[test]
	fn test_close() {
		let client = GymClient::default();
		let env = client.make("FrozenLake-v1");
		env.close();
	}
}
