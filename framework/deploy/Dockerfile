# LearnedFP framework — collection-only Node service.
#
# Build context: the framework/ directory (one level up from deploy/).
#   docker build -f deploy/Dockerfile -t learnedfp-framework .
#
# The framework handles probe generation, client-side rendering, raw RGBA
# upload, ingestion, and validation. It does NOT run identification —
# that lives in the separate evaluation pipeline and requires the
# trained encoders + gallery, neither of which are released.

FROM node:20-bookworm-slim

# Native libs needed by `canvas` and `sharp` (image decoding / glyph
# rasterization on the server side for sample post-processing).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config \
      libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Node deps in their own layer for build caching.
COPY server/package.json server/package-lock.json ./server/
RUN cd server && npm ci --omit=dev

# App source.
COPY server/ ./server/
COPY client/ ./client/
COPY deploy/startup.sh ./startup.sh
RUN chmod +x startup.sh

# PERSIST_DIR is the parent of both per-surface stores:
#   <PERSIST_DIR>/learnedfp/save/<deviceId>/...    (canvas .rgba + samples_index.json + validation.ndjson)
#   <PERSIST_DIR>/audio/save/<deviceId>/...        (audio .f32 + samples_index.json + validation.ndjson)
# Mount a host volume at /data to inspect uploads from the host.
ENV PERSIST_DIR=/data
ENV NODE_ENV=production

EXPOSE 3000
CMD ["./startup.sh"]
