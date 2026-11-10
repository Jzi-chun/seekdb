# MiniVectorDB - round 2 competition image (flat source layout).
#
#   docker build -t minivectordb .
#   docker run --rm -i --network=none --cpus=2 --memory=2g \
#       --mount type=bind,src=/tmp/data,dst=/data minivectordb
#
# The database reads JSON lines from stdin and writes JSON lines to stdout; no
# port is exposed and no background service is started.

FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY engine.h internal.h numparse.h main.cpp engine.cpp sql.cpp ./
RUN mkdir -p /out && g++ -std=c++17 -O2 -DNDEBUG -pthread \
        -o /out/minivectordb main.cpp engine.cpp sql.cpp

FROM ubuntu:24.04

RUN mkdir -p /data && chmod 777 /data
WORKDIR /app
COPY --from=build /out/minivectordb /app/minivectordb

VOLUME ["/data"]
ENTRYPOINT ["/app/minivectordb"]
CMD ["--data-dir", "/data"]
