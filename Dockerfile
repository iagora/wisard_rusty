FROM rust:1.54-buster as builder
WORKDIR /usr/src/actix_wisard
COPY . .
RUN cargo install --bin actix_wisard_u8 --path .

FROM debian:buster-slim
#RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY --from=builder /usr/local/cargo/bin/actix_wisard_u8 /usr/local/bin/actix_wisard_u8
CMD ["actix_wisard_u8"]