package main

import (
	"encoding/json"
	"log"
	"net/http"
	"slices"
	"time"
)

type uiSource struct {
	Source
	Enabled bool   `json:"enabled"`
	Stream  string `json:"stream,omitempty"`
	Error   string `json:"error,omitempty"`
}

type uiState struct {
	ConnectorID  string       `json:"connectorId"`
	Name         string       `json:"name"`
	APIURL       string       `json:"apiUrl"`
	APIKeySource string       `json:"apiKeySource"` // flag | config | none
	FilesDir     string       `json:"filesDir,omitempty"`
	Health       HealthStatus `json:"health"`
	Sources      []uiSource   `json:"sources"`
}

func startUI(app *App, addr string) {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(uiHTML))
	})

	mux.HandleFunc("GET /api/state", func(w http.ResponseWriter, r *http.Request) {
		streams := app.mgr.States()
		streamBySource := map[string]StreamState{}
		for _, s := range streams {
			streamBySource[s.SourceLocalID] = s
		}
		app.mu.Lock()
		_, keySource := app.apiKey()
		state := uiState{
			ConnectorID:  app.ID,
			Name:         app.Name,
			APIURL:       app.APIURL,
			APIKeySource: keySource,
			FilesDir:     app.FilesDir,
			Health:       app.health,
		}
		for _, s := range app.sources {
			us := uiSource{Source: s, Enabled: !slices.Contains(app.cfg.Disabled, s.LocalID)}
			if st, ok := streamBySource[s.LocalID]; ok {
				us.Stream = st.State
				us.Error = st.Error
			}
			state.Sources = append(state.Sources, us)
		}
		app.mu.Unlock()
		writeJSON(w, state)
	})

	mux.HandleFunc("POST /api/config", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			APIKey string `json:"apiKey"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.APIKey == "" {
			http.Error(w, "apiKey required", http.StatusBadRequest)
			return
		}
		if err := app.SetAPIKey(body.APIKey); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /api/rtsp", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Name string `json:"name"`
			URL  string `json:"url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid body", http.StatusBadRequest)
			return
		}
		if err := app.AddRTSP(body.Name, body.URL); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /api/toggle", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			LocalID string `json:"localId"`
			Enabled bool   `json:"enabled"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.LocalID == "" {
			http.Error(w, "localId required", http.StatusBadRequest)
			return
		}
		if err := app.SetSourceEnabled(body.LocalID, body.Enabled); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]string{"status": "ok"})
	})

	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		log.Printf("local UI on http://%s", addr)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Printf("local UI stopped: %v", err)
		}
	}()
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

const uiHTML = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roboflow Video Connector</title>
<style>
  :root { --purple:#6706CE; --ink:#1B1327; --mut:#6c6480; --line:#e6e0f0; --bg:#faf9fd; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 system-ui,sans-serif; color:var(--ink); background:var(--bg); }
  .wrap { max-width:760px; margin:0 auto; padding:32px 20px; }
  h1 { font-size:18px; margin:0; }
  h1 small { color:var(--mut); font-weight:400; font-size:12px; margin-left:8px; }
  .status { display:flex; gap:8px; align-items:center; margin:14px 0 24px; font-size:13px; color:var(--mut); flex-wrap:wrap; }
  .dot { width:8px; height:8px; border-radius:99px; background:#bbb; }
  .dot.ok { background:#18a663; } .dot.bad { background:#d33; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin-bottom:16px; }
  .card h2 { font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--mut); margin:0 0 12px; }
  table { width:100%; border-collapse:collapse; font-size:13.5px; }
  td { padding:8px 6px; border-top:1px solid var(--line); }
  td.kind { color:var(--mut); font-family:ui-monospace,monospace; font-size:12px; }
  td.stream { font-family:ui-monospace,monospace; font-size:12px; color:var(--purple); }
  input[type=text], input[type=password] { border:1px solid var(--line); border-radius:6px; padding:7px 10px; font-size:13px; width:100%; }
  .row { display:flex; gap:8px; }
  button { background:var(--purple); color:#fff; border:0; border-radius:6px; padding:7px 14px; font-size:13px; cursor:pointer; white-space:nowrap; }
  button:hover { opacity:.9; }
  .switch { position:relative; width:34px; height:20px; display:inline-block; }
  .switch input { opacity:0; width:0; height:0; }
  .slider { position:absolute; inset:0; background:#ccc; border-radius:99px; transition:.15s; cursor:pointer; }
  .slider:before { content:""; position:absolute; width:16px; height:16px; left:2px; top:2px; background:#fff; border-radius:99px; transition:.15s; }
  input:checked + .slider { background:var(--purple); }
  input:checked + .slider:before { transform:translateX(14px); }
  .muted { color:var(--mut); font-size:12.5px; }
  .err { color:#c22; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🦝 Video Connector <small id="cid"></small></h1>
  <div class="status">
    <span class="dot" id="hdot"></span><span id="hmsg">loading…</span>
  </div>

  <div class="card" id="keycard">
    <h2>Platform connection</h2>
    <div class="muted" style="margin-bottom:8px;">API: <span id="apiurl"></span> · key: <b id="keysrc"></b></div>
    <div class="row">
      <input type="password" id="apikey" placeholder="Roboflow API key">
      <button onclick="saveKey()">Save key</button>
    </div>
  </div>

  <div class="card">
    <h2>Sources</h2>
    <table id="sources"><tbody></tbody></table>
    <div class="muted" id="nosources" style="display:none;">No sources discovered yet.</div>
  </div>

  <div class="card">
    <h2>Add RTSP camera</h2>
    <div class="row">
      <input type="text" id="rtspname" placeholder="Name (e.g. loading-dock)" style="max-width:200px;">
      <input type="text" id="rtspurl" placeholder="rtsp://user:pass@192.168.1.20:554/stream1">
      <button onclick="addRtsp()">Add</button>
    </div>
    <div class="err" id="rtsperr"></div>
  </div>
</div>
<script>
async function post(path, body) {
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  if (!r.ok) throw new Error(await r.text());
}
async function saveKey() {
  const v = document.getElementById("apikey").value.trim();
  if (!v) return;
  await post("/api/config", {apiKey:v});
  document.getElementById("apikey").value = "";
  load();
}
async function addRtsp() {
  const errEl = document.getElementById("rtsperr");
  errEl.textContent = "";
  try {
    await post("/api/rtsp", {name:document.getElementById("rtspname").value, url:document.getElementById("rtspurl").value});
    document.getElementById("rtspname").value = ""; document.getElementById("rtspurl").value = "";
    load();
  } catch (e) { errEl.textContent = e.message; }
}
async function toggle(localId, enabled) {
  await post("/api/toggle", {localId, enabled});
  load();
}
function esc(s){ const d=document.createElement("div"); d.textContent=s||""; return d.innerHTML; }
async function load() {
  const s = await (await fetch("/api/state")).json();
  document.getElementById("cid").textContent = s.connectorId;
  document.getElementById("apiurl").textContent = s.apiUrl;
  document.getElementById("keysrc").textContent = s.apiKeySource === "none" ? "not set" : "set via " + s.apiKeySource;
  const dot = document.getElementById("hdot"), msg = document.getElementById("hmsg");
  if (s.apiKeySource === "none") { dot.className="dot"; msg.textContent="waiting for API key — sources are not registered yet"; }
  else if (s.health && s.health.ok) { dot.className="dot ok"; msg.textContent="connected to platform · last poll " + new Date(s.health.at).toLocaleTimeString(); }
  else if (s.health && s.health.error) { dot.className="dot bad"; msg.textContent="platform error: " + s.health.error; }
  else { dot.className="dot"; msg.textContent="connecting…"; }
  const tbody = document.querySelector("#sources tbody");
  tbody.innerHTML = "";
  (s.sources||[]).forEach(src => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      '<td class="kind">' + esc(src.kind) + '</td>' +
      '<td>' + esc(src.label) + (src.error ? '<div class="err">' + esc(src.error) + '</div>' : '') + '</td>' +
      '<td class="stream">' + esc(src.stream || "") + '</td>' +
      '<td style="text-align:right;"><label class="switch"><input type="checkbox" ' + (src.enabled ? "checked" : "") + '><span class="slider"></span></label></td>';
    tr.querySelector("input").addEventListener("change", (e) => toggle(src.localId, e.target.checked));
    tbody.appendChild(tr);
  });
  document.getElementById("nosources").style.display = (s.sources||[]).length ? "none" : "block";
}
load(); setInterval(load, 2000);
</script>
</body>
</html>`
