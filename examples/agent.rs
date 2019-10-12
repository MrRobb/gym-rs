extern crate gym;
extern crate rand;

use gym::GymClient;
use rand::Rng;

// Hyperparameters
const ALPHA: f64 = 0.1;
const GAMMA: f64 = 0.6;
const EPSILON: f64 = 0.1;

fn main() {

    let mut rng = rand::thread_rng();
	let client = GymClient::new("127.0.0.1".parse().unwrap(), 8000).unwrap();
	let env = client.make("Taxi-v3".into(), None).expect("Could not make environment");
    let mut qtable = [[0.0; 6]; 500];

    // Exploration
	for ep in 0..100_000 {

        let mut done = false;

		let obs = env.reset().unwrap();
        let mut i_state = obs[0] as usize;

		while !done {

            let action = if rng.gen_bool(EPSILON) {
                env.action_space().sample()
            }
            else {
                vec!(qtable[i_state].iter().cloned().fold(-1./0. /* -inf */, f64::max))
            };
            let i_action: usize = action[0] as usize;

            let old_value = qtable[i_state][i_action];
			let state = env.step(action, false).unwrap();
            let next_max = qtable[i_state].iter().cloned().fold(-1./0. /* -inf */, f64::max);
            let next_value = (1.0 - ALPHA) * old_value + ALPHA * (state.reward + GAMMA * next_max);
            qtable[i_state][i_action] = next_value;

            i_state = state.observation[0] as usize;
			done = state.done;
		}

        //let zeros: usize = qtable.iter().map(|&x| x.iter().filter(|&&x| x == 0).count()).sum();
        if ep % 100 == 0 {
            println!("Finished episode {}", ep);
        }
    }
}