#!/usr/bin/env zsh

# 如果 wasm-pack 不存在，由  cargo 安装
if ! command -v wasm-pack &> /dev/null
then
  echo "wasm-pack not found, installing..."
  cargo install wasm-pack &&　cargo install wasm-bindgen-cli
fi

# ubuntu 下有一个bug，就是wasm－opt 有问题，需要关闭 --no-opt

RUSTUP_DIST_SERVER="https://rsproxy.cn" && \
    RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup" && \
    RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web -d  ../ochw-web/pkg --release --no-opt