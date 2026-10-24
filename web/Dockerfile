# Next.js standalone build.
FROM node:22-alpine AS build
WORKDIR /web
RUN corepack enable && corepack prepare pnpm@12.15.0 --activate
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM node:22-alpine AS run
WORKDIR /web
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
RUN addgroup -S app && adduser -S app -G app
COPY --from=build --chown=app:app /web/.next/standalone ./
COPY --from=build --chown=app:app /web/.next/static ./.next/static
COPY --from=build --chown=app:app /web/public ./public
USER app
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s CMD wget -qO- http://127.0.0.1:3000/api/session >/dev/null || exit 1
CMD ["node", "server.js"]
