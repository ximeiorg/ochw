#!/usr/bin/env zsh

RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web