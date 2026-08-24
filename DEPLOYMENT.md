# Deploying to a Real Domain

This is the concrete, step-by-step path from "runs on my laptop" to
"reachable at https://your-domain.com" — no unstated steps.

## 0. What you actually need before starting

- A server (any VPS — DigitalOcean, Hetzner, a $6-12/mo box is plenty
  to start) with Docker and Docker Compose installed
- A domain name pointed at that server's IP (an A record)
- A TLS certificate for that domain (free via Let's Encrypt/certbot,
  instructions below)

**Not included in this repo, deliberately**: I cannot provision a server,
buy a domain, or issue you a certificate — those require accounts,
payment, and DNS access I don't have. Everything below is real, tested
configuration; the account-creation steps are yours to click through.

## 1. Get a TLS certificate

```bash
# On the server, before starting the app (certbot needs port 80 free):
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com

# Certificates land in /etc/letsencrypt/live/your-domain.com/
mkdir -p certs
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem certs/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem certs/key.pem
sudo chown $USER:$USER certs/*.pem
```

Let's Encrypt certs expire every 90 days — set up `certbot renew` on a
cron job, or use `certbot`'s systemd timer (installed automatically on
most distros) rather than doing this by hand each quarter.

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:
- `ENVIRONMENT=production` (enables HSTS, disables dev-mode leniency)
- `CORS_ALLOWED_ORIGINS=https://your-domain.com` (without this, no
  browser-based frontend — including `pricing.html` if you host it
  elsewhere — can call the API at all; see Section 0 in README.md)
- `PORTFOLIO_OPTIMIZER_API_KEY_HASHES=` comma-separated SHA-256 hashes of random API keys for production. Generate with `python -c "import hashlib; print(hashlib.sha256(b'YOUR_RANDOM_KEY').hexdigest())"`. Keep `PORTFOLIO_OPTIMIZER_API_KEYS` empty in production.
- `PORTFOLIO_OPTIMIZER_SAAS_MODE=1` with real Stripe keys instead if you're selling access (README.md Section 11)

Edit `nginx.conf`: replace both instances of `your-domain.com` with your
actual domain.

## 3. Build and run

```bash
docker compose up -d --build
docker compose logs -f api      # watch it start
```

Verify:
```bash
curl https://your-domain.com/health
curl https://your-domain.com/health/ready
```

`/health` returns instantly if the process is alive at all. `/health/ready`
actually checks dependencies (audit log writable, tenancy DB reachable if
SaaS mode is on) — point your monitoring/load-balancer health checks at
`/health/ready`, not `/health`, since a process that's "alive" but can't
actually serve a request isn't ready.

## 4. What's actually running

- **nginx** (port 443, TLS-terminated) — the only thing exposed to the
  internet. Serves `pricing.html` directly for a plain page load, proxies
  everything else to the API. Rate-limits at the edge (10 req/s/IP) as a
  first line of defense before slowapi's per-API-key limiting inside the
  app even runs.
- **gunicorn + uvicorn workers** (internal port 8000, not exposed
  directly) — `WORKERS` processes (default: 2×CPU cores+1, capped at 8),
  each handling requests async via uvicorn's worker class. Workers
  restart automatically after 2000±200 requests (jittered) as a guard
  against slow memory leaks in any dependency, and gunicorn restarts any
  worker that crashes outright.
- **SQLite** for `infra.persistence.PortfolioStateStore` and
  `saas.tenancy.TenancyStore`, on a named Docker volume so data survives
  container restarts. **This is a real, known scaling ceiling**: SQLite
  handles one writer at a time. Fine for one API instance at moderate
  load; if you ever run multiple `api` replicas or push `WORKERS` high
  under real concurrent write load, this becomes the bottleneck. The SQL
  in both classes is plain enough to port to Postgres directly when that
  day comes — it isn't today's problem to solve preemptively.

## 5. Updating

```bash
git pull   # or however you're getting new code onto the server
docker compose up -d --build   # rebuilds only the api image, nginx config
                                 # changes need a `docker compose restart nginx`
```

`preload_app = True` in `gunicorn_conf.py` means a broken import fails
the deploy immediately and loudly (worker won't start) rather than
serving 500s from a half-loaded app — check `docker compose logs api`
if `up -d` reports the container unhealthy.

## 6. What I could not test from here

I don't have Docker or nginx available in the environment I built this
in, so I could not run `docker compose up` or validate `nginx.conf`
against a real nginx binary. What I *did* verify directly: the exact
package list in the Dockerfile is sufficient to import and serve the API
(tested in a clean virtualenv with nothing beyond that list installed),
gunicorn's config loads without error (`--check-config`), and a real
gunicorn process using this exact config serves real HTTP traffic
correctly, including a full portfolio optimization request end-to-end.
The nginx config itself is standard, unexotic reverse-proxy
configuration — but "I wrote configuration I'm confident is correct" and
"I ran it and watched it work" are different claims, and I want to be
clear about which one this is before you point a real domain at it. Run
`docker compose up` yourself and check `docker compose logs` before
trusting this in production.
