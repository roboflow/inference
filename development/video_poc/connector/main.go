// Roboflow video connector (POC): runs on a customer network, discovers local
// video sources (USB cameras, RTSP cameras, video files), registers them with
// the Roboflow platform via an outbound-only polling control channel, and
// pushes video to a cloud ingest endpoint only when commanded to.
//
// No inbound ports are required for the platform connection. A small local web
// UI (default http://127.0.0.1:8070) shows status and allows runtime config:
// set the API key, enable/disable sources, add RTSP cameras.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"
)

type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

type HealthStatus struct {
	At       time.Time `json:"at"`
	OK       bool      `json:"ok"`
	Error    string    `json:"error,omitempty"`
	Commands int       `json:"commands"`
}

type App struct {
	mu sync.Mutex

	// immutable after start
	ID       string
	Name     string
	Hostname string
	APIURL   string
	FilesDir string
	Ffmpeg   string
	cfgPath  string
	flagKey  string
	flagRTSP []string

	cfg     FileConfig
	sources []Source
	mgr     *StreamManager
	disco   *Discovery
	health  HealthStatus
}

func (a *App) apiKey() (key, source string) {
	if a.flagKey != "" {
		return a.flagKey, "flag"
	}
	if a.cfg.APIKey != "" {
		return a.cfg.APIKey, "config"
	}
	return "", "none"
}

func (a *App) client() *APIClient {
	key, _ := a.apiKey()
	if key == "" {
		return nil
	}
	return &APIClient{BaseURL: strings.TrimRight(a.APIURL, "/"), APIKey: key, ConnectorID: a.ID}
}

func (a *App) rediscover() {
	a.mu.Lock()
	a.disco.RTSPFlags = append(append([]string{}, a.flagRTSP...), a.cfg.RTSP...)
	a.mu.Unlock()
	sources := a.disco.Discover()
	a.mu.Lock()
	a.sources = sources
	a.mu.Unlock()
}

func (a *App) enabledSources() []Source {
	a.mu.Lock()
	defer a.mu.Unlock()
	var out []Source
	for _, s := range a.sources {
		if !slices.Contains(a.cfg.Disabled, s.LocalID) {
			out = append(out, s)
		}
	}
	return out
}

func (a *App) SetAPIKey(key string) error {
	a.mu.Lock()
	a.cfg.APIKey = strings.TrimSpace(key)
	cfg := a.cfg
	path := a.cfgPath
	a.mu.Unlock()
	log.Print("api key updated via local UI")
	return saveConfig(path, cfg)
}

func (a *App) AddRTSP(name, url string) error {
	name = strings.TrimSpace(name)
	url = strings.TrimSpace(url)
	if url == "" || !strings.Contains(url, "://") {
		return fmt.Errorf("a valid rtsp:// URL is required")
	}
	if name == "" {
		name = url
	}
	entry := name + "=" + url
	a.mu.Lock()
	a.cfg.RTSP = append(a.cfg.RTSP, entry)
	cfg := a.cfg
	path := a.cfgPath
	a.mu.Unlock()
	if err := saveConfig(path, cfg); err != nil {
		return err
	}
	log.Printf("rtsp source added via local UI: %s", entry)
	a.rediscover()
	return nil
}

func (a *App) SetSourceEnabled(localID string, enabled bool) error {
	a.mu.Lock()
	idx := slices.Index(a.cfg.Disabled, localID)
	if enabled && idx >= 0 {
		a.cfg.Disabled = slices.Delete(a.cfg.Disabled, idx, idx+1)
	} else if !enabled && idx < 0 {
		a.cfg.Disabled = append(a.cfg.Disabled, localID)
	}
	cfg := a.cfg
	path := a.cfgPath
	a.mu.Unlock()
	if !enabled {
		a.mgr.Stop(localID) // stop pushing immediately; platform reconciles on next poll
	}
	log.Printf("source %s enabled=%v via local UI", localID, enabled)
	return saveConfig(path, cfg)
}

func (a *App) setHealth(h HealthStatus) {
	a.mu.Lock()
	a.health = h
	a.mu.Unlock()
}

func main() {
	var rtspFlags stringList
	apiURL := flag.String("api-url", os.Getenv("RF_API_URL"), "Roboflow API base URL")
	apiKey := flag.String("api-key", os.Getenv("ROBOFLOW_API_KEY"), "Roboflow API key (can also be set in the local UI)")
	id := flag.String("id", "", "connector id (default: conn-<hostname>)")
	name := flag.String("name", "", "display name (default: hostname)")
	filesDir := flag.String("files-dir", "", "directory of video files to expose as sources")
	ffmpegPath := flag.String("ffmpeg", "", "path to ffmpeg (default: $PATH, then ./bin/ffmpeg)")
	pollInterval := flag.Duration("poll", 2*time.Second, "healthcheck poll interval")
	uiAddr := flag.String("ui-addr", "127.0.0.1:8070", "local web UI address (empty to disable)")
	cfgPath := flag.String("config", "connector.json", "path to the runtime config file")
	flag.Var(&rtspFlags, "rtsp", "RTSP source as name=url or url (repeatable)")
	flag.Parse()

	if *apiURL == "" {
		log.Fatal("--api-url is required (or RF_API_URL env)")
	}

	hostname, _ := os.Hostname()
	if *id == "" {
		*id = "conn-" + hostname
	}
	if *name == "" {
		*name = hostname
	}

	ffmpeg, err := resolveFFmpeg(*ffmpegPath)
	if err != nil {
		log.Fatalf("ffmpeg not found: %v", err)
	}

	app := &App{
		ID:       *id,
		Name:     *name,
		Hostname: hostname,
		APIURL:   *apiURL,
		FilesDir: *filesDir,
		Ffmpeg:   ffmpeg,
		cfgPath:  *cfgPath,
		flagKey:  *apiKey,
		flagRTSP: rtspFlags,
		cfg:      loadConfig(*cfgPath),
		mgr:      NewStreamManager(ffmpeg),
		disco:    &Discovery{FilesDir: *filesDir, FFmpeg: ffmpeg},
	}

	log.Printf("connector %s starting (ffmpeg: %s)", app.ID, ffmpeg)
	app.rediscover()
	app.mu.Lock()
	logSources(app.sources)
	app.mu.Unlock()

	if *uiAddr != "" {
		startUI(app, *uiAddr)
	}
	if _, src := app.apiKey(); src == "none" {
		log.Printf("no API key yet — set one in the local UI at http://%s", *uiAddr)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	ticker := time.NewTicker(*pollInterval)
	defer ticker.Stop()
	lastDiscovery := time.Now()

	for {
		select {
		case <-ctx.Done():
			log.Print("shutting down: stopping streams, telling the platform goodbye")
			app.mgr.StopAll()
			// explicit offline beats the platform waiting out the lastSeen
			// window; best-effort — a dead network changes nothing here
			if client := app.client(); client != nil {
				if err := client.Goodbye(app.Name, app.Hostname); err != nil {
					log.Printf("goodbye failed (platform will notice via lastSeen): %v", err)
				} else {
					log.Print("platform acknowledged shutdown")
				}
			}
			return
		case <-ticker.C:
		}

		if time.Since(lastDiscovery) > 60*time.Second {
			app.rediscover()
			lastDiscovery = time.Now()
		}

		client := app.client()
		if client == nil {
			continue
		}

		resp, err := client.Healthcheck(HealthcheckRequest{
			Name:     app.Name,
			Hostname: app.Hostname,
			Platform: platformString(),
			Sources:  app.enabledSources(),
			Streams:  app.mgr.States(),
		})
		if err != nil {
			log.Printf("healthcheck failed: %v", err)
			app.setHealth(HealthStatus{At: time.Now(), OK: false, Error: err.Error()})
			continue
		}
		app.setHealth(HealthStatus{At: time.Now(), OK: true, Commands: len(resp.Commands)})

		for _, cmd := range resp.Commands {
			handleCommand(client, app.mgr, app.enabledSources(), cmd)
		}
	}
}

func handleCommand(client *APIClient, mgr *StreamManager, sources []Source, cmd Command) {
	log.Printf("command %s: %s %v", cmd.ID, cmd.Action, cmd.Data)
	var err error
	switch cmd.Action {
	case "start_stream":
		localID, _ := cmd.Data["sourceLocalId"].(string)
		ingestURL, _ := cmd.Data["ingestUrl"].(string)
		if localID == "" || ingestURL == "" {
			err = fmt.Errorf("start_stream requires sourceLocalId and ingestUrl")
			break
		}
		src, found := findSource(sources, localID)
		if !found {
			err = fmt.Errorf("unknown source %q", localID)
			break
		}
		err = mgr.Start(src, ingestURL)
	case "stop_stream":
		localID, _ := cmd.Data["sourceLocalId"].(string)
		if reason, _ := cmd.Data["reason"].(string); reason != "" {
			log.Printf("stream %s stop requested by platform: %s", localID, reason)
		}
		mgr.Stop(localID)
	default:
		err = fmt.Errorf("unknown action %q", cmd.Action)
	}

	if ackErr := client.Ack(cmd.ID, err); ackErr != nil {
		log.Printf("ack for %s failed: %v", cmd.ID, ackErr)
	}
	if err != nil {
		log.Printf("command %s failed: %v", cmd.ID, err)
	}
}

func findSource(sources []Source, localID string) (Source, bool) {
	for _, s := range sources {
		if s.LocalID == localID {
			return s, true
		}
	}
	return Source{}, false
}

func resolveFFmpeg(explicit string) (string, error) {
	if explicit != "" {
		return exec.LookPath(explicit)
	}
	if p, err := exec.LookPath("ffmpeg"); err == nil {
		return p, nil
	}
	exe, err := os.Executable()
	if err == nil {
		for _, candidate := range []string{
			filepath.Join(filepath.Dir(exe), "ffmpeg"),
			filepath.Join(filepath.Dir(exe), "bin", "ffmpeg"),
			filepath.Join(filepath.Dir(exe), "..", "bin", "ffmpeg"),
		} {
			if _, statErr := os.Stat(candidate); statErr == nil {
				return candidate, nil
			}
		}
	}
	return "", fmt.Errorf("ffmpeg not on PATH and no bundled copy found")
}

func logSources(sources []Source) {
	log.Printf("discovered %d sources:", len(sources))
	for _, s := range sources {
		log.Printf("  [%s] %s (%s)", s.Kind, s.Label, s.LocalID)
	}
}

func platformString() string {
	return runtime.GOOS + "/" + runtime.GOARCH
}
