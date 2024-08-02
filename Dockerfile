FROM rust:1.80

WORKDIR /usr/src/structured-gen-rust
COPY . .

RUN cargo install --path .

ENTRYPOINT ["structured-gen-rust"]