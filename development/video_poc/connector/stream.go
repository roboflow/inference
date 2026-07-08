package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const maxConsecutiveFailures = 5

// ffmpeg can wedge without exiting — e.g. another app grabs the camera and
// avfoundation silently stops delivering frames. If the progress counters
// stop advancing for this long, the leg is dead and only a restart helps.
const stallTimeout = 15 * time.Second

// cadence of the per-leg "still alive" stats line
const statsLogInterval = 30 * time.Second

type stream struct {
	source    Source
	ingestURL string
	cancel    context.CancelFunc
	done      chan struct{}

	mu      sync.Mutex
	state   string
	lastErr string
}

type StreamManager struct {
	mu      sync.Mutex
	ffmpeg  string
	streams map[string]*stream
}

func NewStreamManager(ffmpeg string) *StreamManager {
	return &StreamManager{ffmpeg: ffmpeg, streams: map[string]*stream{}}
}

func (m *StreamManager) Start(src Source, ingestURL string) error {
	m.Stop(src.LocalID)

	ctx, cancel := context.WithCancel(context.Background())
	s := &stream{
		source:    src,
		ingestURL: ingestURL,
		cancel:    cancel,
		done:      make(chan struct{}),
		state:     "starting",
	}
	m.mu.Lock()
	m.streams[src.LocalID] = s
	m.mu.Unlock()

	go m.supervise(ctx, s)
	log.Printf("stream %s -> %s started", src.LocalID, ingestURL)
	return nil
}

func (m *StreamManager) supervise(ctx context.Context, s *stream) {
	defer close(s.done)
	failures := 0
	flexible := false
	for {
		args := ffmpegArgs(s.source, s.ingestURL, flexible)
		cmd := exec.CommandContext(ctx, m.ffmpeg, args...)
		var stderrTail tailBuffer
		cmd.Stderr = &stderrTail
		stdout, pipeErr := cmd.StdoutPipe()

		start := time.Now()
		var err error
		stalled := false
		if pipeErr != nil {
			err = pipeErr
		} else if err = cmd.Start(); err == nil {
			watchCtx, stopWatch := context.WithCancel(ctx)
			stallCh := make(chan struct{}, 1)
			go m.watchProgress(watchCtx, s, stdout, stallCh)
			waitCh := make(chan error, 1)
			go func() { waitCh <- cmd.Wait() }()
			select {
			case err = <-waitCh:
			case <-stallCh:
				stalled = true
				log.Printf("stream %s STALLED (no frames for %s — camera grabbed by another app?), restarting ffmpeg",
					s.source.LocalID, stallTimeout)
				_ = cmd.Process.Kill()
				err = <-waitCh
			}
			stopWatch()
		}
		if ctx.Err() != nil {
			return
		}
		ranAWhile := time.Since(start) > 30*time.Second
		if stalled || ranAWhile {
			failures = 0 // it ran for a while; treat exit as a hiccup, not a config problem
		} else {
			failures++
		}
		if s.source.Kind == "usb" {
			if (stalled || !ranAWhile) && !flexible {
				// device likely shared at a different format now — reopen
				// format-agnostic and normalize instead of fighting for
				// exactly 1280x720@30
				flexible = true
				log.Printf("stream %s switching to camera-sharing mode (any format, normalized to 1280x720@30)",
					s.source.LocalID)
			} else if ranAWhile && !stalled && flexible {
				// a clean long run ended (relay hiccup etc.) — try native
				// format again; if the camera is still shared we'll be back
				flexible = false
			}
		}
		s.mu.Lock()
		s.lastErr = fmt.Sprintf("ffmpeg exited: %v; %s", err, stderrTail.Tail())
		if failures >= maxConsecutiveFailures {
			s.state = "error"
			s.mu.Unlock()
			log.Printf("stream %s giving up after %d failures: %s", s.source.LocalID, failures, s.lastErr)
			return
		}
		s.state = "starting"
		s.mu.Unlock()
		if !stalled {
			log.Printf("stream %s ffmpeg exited (%v), restarting", s.source.LocalID, err)
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
}

// watchProgress consumes ffmpeg's -progress output (key=value blocks) and
// fires stallCh if the frame/time counters stop advancing. It also promotes
// the leg to "running" on first real progress (a positive signal, unlike the
// process-stayed-up heuristic) and logs a periodic stats line.
func (m *StreamManager) watchProgress(ctx context.Context, s *stream, r io.Reader, stallCh chan struct{}) {
	type progress struct{ frame, timeUS int64 }
	lines := make(chan progress, 8)
	go func() {
		defer close(lines)
		var cur progress
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			key, val, found := strings.Cut(strings.TrimSpace(sc.Text()), "=")
			if !found {
				continue
			}
			switch key {
			case "frame":
				cur.frame, _ = strconv.ParseInt(val, 10, 64)
			case "out_time_us":
				cur.timeUS, _ = strconv.ParseInt(val, 10, 64)
			case "progress": // end of a block
				select {
				case lines <- cur:
				default:
				}
			}
		}
	}()

	started := time.Now()
	last := progress{-1, -1}
	lastAdvance := time.Now()
	lastLog := time.Now()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case p, ok := <-lines:
			if !ok {
				return // ffmpeg exited; supervise handles it
			}
			// -c copy legs don't count frames, so time progress counts too
			if p.frame > last.frame || p.timeUS > last.timeUS {
				last = p
				lastAdvance = time.Now()
				s.mu.Lock()
				if s.state == "starting" {
					s.state = "running"
					s.mu.Unlock()
					log.Printf("stream %s delivering (first frames after %.1fs)",
						s.source.LocalID, time.Since(started).Seconds())
				} else {
					s.mu.Unlock()
				}
			}
			if time.Since(lastLog) >= statsLogInterval {
				lastLog = time.Now()
				log.Printf("stream %s: frame=%d stream_time=%s uptime=%s",
					s.source.LocalID, last.frame,
					(time.Duration(last.timeUS) * time.Microsecond).Round(time.Second),
					time.Since(started).Round(time.Second))
			}
		case <-ticker.C:
			if time.Since(lastAdvance) > stallTimeout {
				select {
				case stallCh <- struct{}{}:
				default:
				}
				return
			}
		}
	}
}

func (m *StreamManager) markRunningIfAlive(s *stream) {
	// ffmpeg gives no positive signal without parsing progress output; consider the
	// stream running once the process has stayed up past its first few seconds.
	s.mu.Lock()
	if s.state == "starting" {
		s.state = "running"
	}
	s.mu.Unlock()
}

func (m *StreamManager) Stop(localID string) {
	m.mu.Lock()
	s, ok := m.streams[localID]
	if ok {
		delete(m.streams, localID)
	}
	m.mu.Unlock()
	if !ok {
		return
	}
	s.cancel()
	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
	}
	log.Printf("stream %s stopped", localID)
}

func (m *StreamManager) StopAll() {
	m.mu.Lock()
	ids := make([]string, 0, len(m.streams))
	for id := range m.streams {
		ids = append(ids, id)
	}
	m.mu.Unlock()
	for _, id := range ids {
		m.Stop(id)
	}
}

func (m *StreamManager) States() []StreamState {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]StreamState, 0, len(m.streams))
	for _, s := range m.streams {
		m.markRunningIfAlive(s)
		s.mu.Lock()
		st := StreamState{SourceLocalID: s.source.LocalID, State: s.state}
		if s.state == "error" {
			st.Error = s.lastErr
		}
		s.mu.Unlock()
		out = append(out, st)
	}
	return out
}

func encodeArgs() []string {
	return []string{
		"-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
		"-pix_fmt", "yuv420p", "-g", "30", "-bf", "0", "-an",
	}
}

func outputArgs(ingestURL string) []string {
	// -progress pipe:1 feeds the stall watchdog and stats (see watchProgress)
	return []string{"-progress", "pipe:1", "-f", "rtsp", "-rtsp_transport", "tcp", ingestURL}
}

func ffmpegArgs(src Source, ingestURL string, flexible bool) []string {
	var args []string
	switch src.Kind {
	case "file":
		args = []string{"-hide_banner", "-loglevel", "warning", "-re", "-stream_loop", "-1", "-i", src.Path}
		args = append(args, encodeArgs()...)
	case "rtsp":
		// passthrough: no decode, no encode — just remux
		args = []string{"-hide_banner", "-loglevel", "warning", "-rtsp_transport", "tcp", "-i", src.URL, "-c", "copy", "-an"}
	case "usb":
		if flexible {
			// camera-sharing mode: another app renegotiated the device
			// format (macOS shares cameras, but at ONE format) and the
			// strict 1280x720@30 open stalls or fails while it holds the
			// device. Take whatever format is active and normalize in the
			// filter graph so downstream never notices.
			if runtime.GOOS == "darwin" {
				args = []string{
					"-hide_banner", "-loglevel", "warning",
					"-f", "avfoundation",
					"-i", src.Device + ":none",
				}
			} else {
				args = []string{
					"-hide_banner", "-loglevel", "warning",
					"-f", "v4l2",
					"-i", src.Device,
				}
			}
			args = append(args, "-vf", "fps=30,scale=1280:720")
		} else if runtime.GOOS == "darwin" {
			args = []string{
				"-hide_banner", "-loglevel", "warning",
				"-f", "avfoundation", "-framerate", "30", "-video_size", "1280x720",
				"-i", src.Device + ":none",
			}
		} else {
			args = []string{
				"-hide_banner", "-loglevel", "warning",
				"-f", "v4l2", "-framerate", "30", "-video_size", "1280x720",
				"-i", src.Device,
			}
		}
		args = append(args, encodeArgs()...)
	}
	return append(args, outputArgs(ingestURL)...)
}

// tailBuffer keeps the last ~2KB of writes for error reporting.
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > 2048 {
		t.buf = t.buf[len(t.buf)-2048:]
	}
	return len(p), nil
}

func (t *tailBuffer) Tail() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	lines := strings.Split(strings.TrimSpace(string(t.buf)), "\n")
	if len(lines) > 3 {
		lines = lines[len(lines)-3:]
	}
	return strings.Join(lines, " | ")
}
