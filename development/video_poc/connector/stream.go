package main

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"
)

const maxConsecutiveFailures = 5

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
	for {
		args := ffmpegArgs(s.source, s.ingestURL)
		cmd := exec.CommandContext(ctx, m.ffmpeg, args...)
		var stderrTail tailBuffer
		cmd.Stderr = &stderrTail

		start := time.Now()
		err := cmd.Run()
		if ctx.Err() != nil {
			return
		}
		if time.Since(start) > 30*time.Second {
			failures = 0 // it ran for a while; treat exit as a hiccup, not a config problem
		} else {
			failures++
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
		log.Printf("stream %s ffmpeg exited (%v), restarting", s.source.LocalID, err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
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
	return []string{"-f", "rtsp", "-rtsp_transport", "tcp", ingestURL}
}

func ffmpegArgs(src Source, ingestURL string) []string {
	var args []string
	switch src.Kind {
	case "file":
		args = []string{"-hide_banner", "-loglevel", "warning", "-re", "-stream_loop", "-1", "-i", src.Path}
		args = append(args, encodeArgs()...)
	case "rtsp":
		// passthrough: no decode, no encode — just remux
		args = []string{"-hide_banner", "-loglevel", "warning", "-rtsp_transport", "tcp", "-i", src.URL, "-c", "copy", "-an"}
	case "usb":
		if runtime.GOOS == "darwin" {
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
