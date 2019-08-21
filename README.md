# gym-rs

![HitCount](http://hits.dwyl.io/mrrobb/gym-rs.svg)
[![Crates.io](https://img.shields.io/crates/v/gym)](https://crates.io/crates/gym)
[![Docs.rs](https://docs.rs/gym/badge.svg)](https://docs.rs/gym/latest/gym)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/MrRobb/gym-rs/blob/master/LICENSE)

OpenAI gym binding for Rust.

> Actively maintained! If you have any problem just [create an issue](https://github.com/MrRobb/gym-rs/issues/new).

### Install

Just install the requierements layed out in the [requirements.txt](https://github.com/MrRobb/gym-rs/blob/master/requirements.txt). 

> If you don't have python installed, go [here](https://realpython.com/installing-python/#windows)

```sh
curl "https://raw.githubusercontent.com/MrRobb/gym-rs/master/requirements.txt" > requirements.txt
pip install -r requirements.txt
```

> If you have any problem trying to install the dependencies --> [create an issue](https://github.com/MrRobb/gym-rs/issues/new).

### Usage

Once everything is installed, just add this crate to your your Rust project.

```toml
# Cargo.toml

[dependencies]
gym = "*"
```

Then, to use it just do:

```rust
// main.rs

extern crate gym;

/* ... */
```

If you have any problem with your installation --> [just create a new issue](https://github.com/MrRobb/gym-rs/issues/new)

### Example

Once you have installed correctly the library, the only thing left is to test if its working ok. To do so, you just have to execute the following commands:

> If you don't have Rust installed go [here](https://www.rust-lang.org/tools/install)

```shell script
git clone https://github.com/MrRobb/gym-rs.git
cd gym-rs
pip install -r requirements.txt
cargo run --example basic
```

> If you have any problem try to install --> [create an issue](https://github.com/MrRobb/gym-rs/issues/new).

> This repository is inspired in [this genius PR](https://github.com/openai/gym-http-api/pull/56) made by @NivenT. It includes changes, but it is based on his contribution.
