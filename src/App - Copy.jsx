# =========================
# Build (keep your Vite defaults)
# =========================
[build]
  command = "npm run build"
  publish = "dist"

# =========================
# PRODUCTION WEBHOOKS (camera)
# Put specific routes FIRST so they win over the wildcard
# =========================
[[redirects]]
  from = "/api/n8n/start"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook/camera/start"
  status = 200
  force  = true

[[redirects]]
  from = "/api/n8n/snapshot"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook/camera/snapshot"
  status = 200
  force  = true

[[redirects]]
  from = "/api/n8n/stop"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook/camera/stop"
  status = 200
  force  = true

# Catch-all PRODUCTION proxy for any other n8n webhook
# e.g. /api/n8n/agent-request → /webhook/agent-request
[[redirects]]
  from = "/api/n8n/*"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook/:splat"
  status = 200
  force  = true

# =========================
# TEST WEBHOOKS (optional)
# If you also call /api/n8n-test/start|snapshot|stop, keep these.
# Otherwise you can remove the specific ones and just keep the wildcard.
# =========================
[[redirects]]
  from = "/api/n8n-test/start"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook-test/camera/start"
  status = 200
  force  = true

[[redirects]]
  from = "/api/n8n-test/snapshot"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook-test/camera/snapshot"
  status = 200
  force  = true

[[redirects]]
  from = "/api/n8n-test/stop"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook-test/camera/stop"
  status = 200
  force  = true

[[redirects]]
  from = "/api/n8n-test/*"
  to   = "https://n8n.srv954455.hstgr.cloud/webhook-test/:splat"
  status = 200
  force  = true

# =========================
# HEADERS
# =========================

# WASM served with correct MIME
[[headers]]
  for = "/*.wasm"
  [headers.values]
    Content-Type = "application/wasm"

# CORS for production proxy (usually not needed because we proxy,
# but harmless if your browser sends preflights)
[[headers]]
  for = "/api/n8n/*"
  [headers.values]
    Access-Control-Allow-Origin  = "*"
    Access-Control-Allow-Headers = "*"
    Access-Control-Allow-Methods = "POST, OPTIONS"

# CORS for test proxy
[[headers]]
  for = "/api/n8n-test/*"
  [headers.values]
    Access-Control-Allow-Origin  = "*"
    Access-Control-Allow-Headers = "*"
    Access-Control-Allow-Methods = "POST, OPTIONS"