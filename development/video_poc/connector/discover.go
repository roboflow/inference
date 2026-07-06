package main

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
)

var videoExtensions = map[string]bool{
	".mp4": true, ".mov": true, ".mkv": true, ".avi": true, ".webm": true, ".ts": true,
}

type Discovery struct {
	FilesDir  string
	RTSPFlags []string
	FFmpeg    string
}

func (d *Discovery) Discover() []Source {
	var sources []Source
	sources = append(sources, d.discoverFiles()...)
	sources = append(sources, d.discoverRTSP()...)
	sources = append(sources, discoverCameras(d.FFmpeg)...)
	sort.Slice(sources, func(i, j int) bool { return sources[i].LocalID < sources[j].LocalID })
	return sources
}

func (d *Discovery) discoverFiles() []Source {
	if d.FilesDir == "" {
		return nil
	}
	entries, err := os.ReadDir(d.FilesDir)
	if err != nil {
		return nil
	}
	var out []Source
	for _, e := range entries {
		if e.IsDir() || !videoExtensions[strings.ToLower(filepath.Ext(e.Name()))] {
			continue
		}
		out = append(out, Source{
			LocalID: "file:" + e.Name(),
			Kind:    "file",
			Label:   e.Name(),
			Path:    filepath.Join(d.FilesDir, e.Name()),
		})
	}
	return out
}

func (d *Discovery) discoverRTSP() []Source {
	var out []Source
	seen := map[string]bool{}
	for _, spec := range d.RTSPFlags {
		label, u := spec, spec
		if name, rest, found := strings.Cut(spec, "="); found && !strings.Contains(name, "://") {
			label, u = name, rest
		}
		localID := "rtsp:" + label
		if seen[localID] {
			continue
		}
		seen[localID] = true
		out = append(out, Source{
			LocalID: localID,
			Kind:    "rtsp",
			Label:   label,
			URL:     u,
		})
	}
	return out
}

func discoverCameras(ffmpeg string) []Source {
	switch runtime.GOOS {
	case "darwin":
		return discoverAVFoundation(ffmpeg)
	case "linux":
		return discoverV4L2()
	default:
		return nil
	}
}

// Parses `ffmpeg -f avfoundation -list_devices true -i ""` stderr, e.g.:
//   [AVFoundation indev @ 0x...] AVFoundation video devices:
//   [AVFoundation indev @ 0x...] [0] FaceTime HD Camera
//   [AVFoundation indev @ 0x...] AVFoundation audio devices:
var avfDeviceRe = regexp.MustCompile(`\[(\d+)\]\s+(.+)$`)

func discoverAVFoundation(ffmpeg string) []Source {
	cmd := exec.Command(ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", "")
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil
	}
	if err := cmd.Start(); err != nil {
		return nil
	}
	var out []Source
	inVideoSection := false
	scanner := bufio.NewScanner(stderr)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "AVFoundation video devices") {
			inVideoSection = true
			continue
		}
		if strings.Contains(line, "AVFoundation audio devices") {
			inVideoSection = false
			continue
		}
		if !inVideoSection {
			continue
		}
		if m := avfDeviceRe.FindStringSubmatch(line); m != nil {
			label := strings.TrimSpace(m[2])
			if strings.Contains(label, "Capture screen") {
				continue
			}
			out = append(out, Source{
				LocalID: "usb:" + m[1],
				Kind:    "usb",
				Label:   label,
				Device:  m[1],
			})
		}
	}
	cmd.Wait() // ffmpeg exits non-zero after listing; that's expected
	return out
}

func discoverV4L2() []Source {
	matches, _ := filepath.Glob("/dev/video*")
	var out []Source
	for _, dev := range matches {
		label := dev
		nameFile := "/sys/class/video4linux/" + filepath.Base(dev) + "/name"
		if b, err := os.ReadFile(nameFile); err == nil {
			label = strings.TrimSpace(string(b))
		}
		out = append(out, Source{
			LocalID: "usb:" + filepath.Base(dev),
			Kind:    "usb",
			Label:   label,
			Device:  dev,
		})
	}
	return out
}

