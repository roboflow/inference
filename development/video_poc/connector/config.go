package main

import (
	"encoding/json"
	"os"
)

// FileConfig holds the connector settings that can be changed at runtime from
// the local web UI. Flags/env take precedence where both are set.
type FileConfig struct {
	APIKey   string   `json:"apiKey,omitempty"`
	RTSP     []string `json:"rtsp,omitempty"`     // "name=url" entries added via the UI
	Disabled []string `json:"disabled,omitempty"` // source localIds excluded from the platform
}

func loadConfig(path string) FileConfig {
	var cfg FileConfig
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg
	}
	_ = json.Unmarshal(data, &cfg)
	return cfg
}

func saveConfig(path string, cfg FileConfig) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}
