# Build context is the repository root (see docker-compose.yml):
#   docker build -f server/Dockerfile .
# Stage 1 builds the React client so the host needs only Docker.
FROM node:22-alpine AS client
ENV npm_config_update_notifier=false
WORKDIR /client
COPY client/package*.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

FROM node:22-alpine
ENV npm_config_update_notifier=false
WORKDIR /app
ENV NODE_ENV=production LEAFCARE_HOST=0.0.0.0 PORT=4000 DATA_DIR=/data CLIENT_DIST=/app/public
COPY server/package*.json ./
RUN npm ci --omit=dev
COPY server/src ./src
COPY --from=client /client/dist ./public
RUN mkdir -p /data && chown node:node /data
USER node
EXPOSE 4000
CMD ["node", "src/index.js"]
