package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

type Source struct {
	LocalID string `json:"localId"`
	Kind    string `json:"kind"` // "usb" | "rtsp" | "file"
	Label   string `json:"label"`
	// kind-specific fields, informational for the platform
	Path   string `json:"path,omitempty"`
	Device string `json:"device,omitempty"`
	URL    string `json:"url,omitempty"`

	// avfoundation enumeration index at discovery time; session-fragile, so
	// only used to disambiguate identically-named cameras (never serialized)
	index string
}

type StreamState struct {
	SourceLocalID string `json:"sourceLocalId"`
	State         string `json:"state"` // "starting" | "running" | "error"
	Error         string `json:"error,omitempty"`
}

type HealthcheckRequest struct {
	Name     string        `json:"name"`
	Hostname string        `json:"hostname"`
	Platform string        `json:"platform"`
	Sources  []Source      `json:"sources"`
	Streams  []StreamState `json:"streams"`
	// final healthcheck of a clean shutdown: the platform marks the
	// connector (and its sources) offline immediately instead of waiting
	// out the lastSeen window
	ShuttingDown bool `json:"shuttingDown,omitempty"`
}

type Command struct {
	ID     string         `json:"id"`
	Action string         `json:"action"`
	Data   map[string]any `json:"data"`
}

type HealthcheckResponse struct {
	Commands []Command `json:"commands"`
}

type APIClient struct {
	BaseURL     string
	APIKey      string
	ConnectorID string
	http        http.Client
}

func (c *APIClient) post(path string, body any, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	u := fmt.Sprintf("%s%s?api_key=%s", c.BaseURL, path, url.QueryEscape(c.APIKey))
	req, err := http.NewRequest(http.MethodPost, u, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	client := c.http
	if client.Timeout == 0 {
		client.Timeout = 10 * time.Second
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("%s returned %s", path, resp.Status)
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func (c *APIClient) Healthcheck(hb HealthcheckRequest) (HealthcheckResponse, error) {
	var resp HealthcheckResponse
	body := struct {
		ConnectorID string `json:"connectorId"`
		HealthcheckRequest
	}{c.ConnectorID, hb}
	err := c.post("/video-connectors/healthcheck", body, &resp)
	return resp, err
}

// Goodbye sends the final shutting-down healthcheck (no sources, no streams).
func (c *APIClient) Goodbye(name, hostname string) error {
	var resp HealthcheckResponse
	body := struct {
		ConnectorID string `json:"connectorId"`
		HealthcheckRequest
	}{c.ConnectorID, HealthcheckRequest{Name: name, Hostname: hostname, ShuttingDown: true}}
	return c.post("/video-connectors/healthcheck", body, &resp)
}

func (c *APIClient) Ack(commandID string, cmdErr error) error {
	body := map[string]any{"success": cmdErr == nil}
	if cmdErr != nil {
		body["error"] = cmdErr.Error()
	}
	path := fmt.Sprintf("/video-connectors/%s/commands/%s/ack", c.ConnectorID, commandID)
	return c.post(path, body, nil)
}
