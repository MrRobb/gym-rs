# gym-rs

![HitCount](http://hits.dwyl.io/mrrobb/gym-rs.svg)
[![Build Status](https://travis-ci.com/MrRobb/gym-rs.svg?branch=master)](https://travis-ci.com/MrRobb/gym-rs)
[![Crates.io](https://img.shields.io/crates/v/gym)](https://crates.io/crates/gym)
[![Docs.rs](https://docs.rs/gym/badge.svg)](https://docs.rs/gym/latest/gym)
[![codecov](https://codecov.io/gh/MrRobb/gym-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/MrRobb/gym-rs)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/MrRobb/gym-rs/blob/master/LICENSE)

OpenAI gym binding for Rust.

> Actively maintained! If you have any problem just [create an issue](https://github.com/MrRobb/gym-rs/issues/new).

### Install

Just install the requierements layed out in the [requirements.txt](https://github.com/MrRobb/gym-rs/blob/master/requirements.txt). 

> If you don't have python installed, go [here](https://realpython.com/installing-python/#windows)

```sh
curl "https://raw.githubusercontent.com/MrRobb/gym-rs/master/requirements.txt" > requirements.txt
pip3 install -r requirements.txt
```

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

### Example

Once you have installed correctly the library, the only thing left is to test if its working ok. To do so, you just have to execute the following commands:

> If you don't have Rust installed go [here](https://www.rust-lang.org/tools/install)

```sh script
git clone https://github.com/MrRobb/gym-rs.git
cd gym-rs
pip3 install -r requirements.txt
cargo run --example basic
```

### Troubleshooting

In Ubuntu 20.04, it is possible that you need to install `swig`. To do that, execute:

```sh
sudo apt-get install swig
```

The example can fail with virtualenv. It's more of a general problem of the cpython crate rather than this one, you can resolve it by setting the PYTHONHOME env var to the module path of the venv, e.g.:

```sh
PYTHONPATH=~/venv-py37/lib/python3.7/site-packages cargo run --example basic
```

## Donation (BTC)

<p align="center">
  <a href="https://i.imgur.com/61OZ7lE.jpg">
    <img src="https://i.imgur.com/61OZ7lE.jpg" width=35%>
	</a>
</p>
<p align="center">BTC address: 3KRM66geiaXWzqs5hRb35dGiQEQAa6JTYU</p>
