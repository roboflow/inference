# Staging video-cell network benchmark plan

Status: design based on read-only inspection on 2026-08-12. Do not run this
load against production.

## Current staging topology and effective limits

The live `crusoe-use1` cell currently routes media as follows:

```text
connector
  -> Crusoe L4 LoadBalancer 204.52.26.16:8554
  -> NodePort 31089 (externalTrafficPolicy=Local)
  -> MediaMTX pod

processor
  -> mediamtx.video-proc.svc:8554
  -> Cilium ClusterIP/VXLAN
  -> MediaMTX pod

browser WHEP signaling
  -> Traefik LoadBalancer
  -> Traefik
  -> MediaMTX :8889

browser WebRTC media
  -> Crusoe L4 LoadBalancer 204.52.26.16:8189/TCP
  -> NodePort 31289
  -> MediaMTX pod
```

The current relay pod is not pinned to a node pool or instance type. At the
time of inspection it was on a shared `c1a.8x` node. Crusoe documents this
instance as having 5 Gbps of VPC bandwidth. A relay restart can move it to a
different node class and silently change the network ceiling. Node exporter
reports a 200 Gbps `ens3` link speed, but that is the virtual NIC link speed,
not the instance's documented VPC entitlement, and must not be used as the
capacity denominator.

The GPU worker was on an `l40s-48gb.10x` node, documented at 175 Gbps VPC
bandwidth. Therefore the relay's current 5 Gbps node is expected to limit the
relay-to-processor path before that GPU node's NIC does.

The three current `pool=benchmark` nodes are `c1a.2x`, documented at 1 Gbps
each. They are not currently viable video load generators: system DaemonSets
reserve about 96% of CPU on a sampled node, and two of the three nodes were at
99-100% CPU during inspection. The immediate consumers were MOFED DaemonSet
containers using approximately 1.84 and 0.99 cores on two-cpu nodes. Even if
that is corrected, the pool can generate
at most roughly 3 Gbps in aggregate before protocol overhead, and reader decode
work will hit its two-vCPU limit much sooner.

Cilium uses kube-proxy replacement, veth datapaths, VXLAN tunnel mode, and MTU
8900. Relay-to-processor traffic is therefore an overlay/east-west path when
the pods are on different nodes. The relay node's Cilium `ct4_global` map was
already at about 6.9% pressure at idle (`295150` entry capacity), while Linux
conntrack was only `1351 / 262144`. Both must be recorded during connection and
reconnection ramps.

Crusoe's public documentation provides per-instance VPC bandwidth but does not
publish a throughput or flow ceiling for the L4 LoadBalancer. The LoadBalancer
is L4 TCP passthrough and source-IP-hash based. Its actual bandwidth, active-flow,
and new-flow knees must be measured, and provider limits should be confirmed
with Crusoe.

References:

- https://docs.crusoecloud.com/compute/virtual-machines/overview/
- https://docs.crusoecloud.com/networking/load-balancers/overview/
- https://docs.crusoecloud.com/networking/load-balancers/load-balancer-metrics/
- https://docs.crusoecloud.com/networking/overview/

## Existing telemetry

The merged MediaMTX monitoring exposes and dashboards:

- ready paths and reader fan-out;
- aggregate and per-hashed-path bytes received/sent;
- RTSP and WebRTC sessions;
- RTSP RTP loss/input errors and WebRTC RTP loss;
- relay CPU, memory, pod network traffic, and restarts.

The cluster Prometheus also already has:

- pod network bytes, packet counts, drops, and errors from cAdvisor;
- physical-node NIC bytes, drops, errors, MTU, and advertised speed;
- TCP retransmissions and Linux conntrack usage from node exporter;
- Cilium forwarded/dropped bytes, BPF map pressure, NAT pressure, and
  node-to-node connectivity latency;
- Hubble flow/drop counters; and
- processor, DCGM, and Kubernetes resource metrics.

The live MediaMTX endpoint exposes additional counters that the current
PodMonitor intentionally drops:

- `rtsp_conns*`;
- RTSP and WebRTC session bytes received/sent;
- RTP packets received/sent and jitter;
- RTCP packets received/sent and input errors.

These should be added to the bounded scrape allowlist before protocol-specific
capacity tests. Path counters are sufficient for total relay bandwidth, but
session counters are required to distinguish RTSP, WHEP/WebRTC, connection
churn, and output-viewer traffic.

Crusoe LoadBalancer and VM telemetry are not present in the cluster Prometheus.
The separate Crusoe metrics API supplies 30-second LoadBalancer counters
(`crusoe_elb_bytes_in`, `crusoe_elb_bytes_out`, packet counts, active flows,
and new flows) and 60-second VM VPC counters
(`crusoe_vm_network_receive_bytes_total` and
`crusoe_vm_network_transmit_bytes_total`). The benchmark runner needs a staging
monitoring token and should query these alongside the higher-resolution
in-cluster metrics.

## Missing measurement capabilities

The current `run_relay_benchmark.py` is a useful smoke harness but is not yet a
capacity-certification harness:

1. Every publisher and reader runs on one host, so it cannot isolate public LB,
   same-node, CNI cross-node, processor-node, or remote-cell paths.
2. Readers decode every stream. That validates frames, but generator CPU can
   become the first bottleneck. Network-only runs need a packet-copy reader;
   a smaller sample of decoding readers should validate media continuity.
3. Process liveness is the only client success signal. The runner needs ffmpeg
   progress, bytes/frames delivered, startup time, reconnect count, and reasoned
   failures.
4. Metrics are scraped as raw endpoints and labels are aggregated away. The
   runner needs Prometheus and Crusoe query-range collectors with run-window
   timestamps and retained labels for the known benchmark resources.
5. The report does not capture Kubernetes image digests, pod/node placement,
   node instance types, documented VPC limits, MediaMTX config hash, Cilium
   mode/version, LoadBalancer ID, or auth/control-plane version.
6. One URL template cannot express multiple reader locations/transports.
7. There are no automatic stop conditions for saturation, loss, or disconnects.
8. Direct high-cardinality relay tests need a safe way to mint short-lived,
   prefix-scoped staging stream credentials. They must not reuse a processor
   fleet secret.

## Recommended safe staging topology

Use a separate performance cell so long-running tests cannot black out the
normal staging preview/processing path:

- namespace `video-proc-bench`;
- its own single MediaMTX origin, ClusterIP, L4 LoadBalancer, DNS names, metrics,
  and short-lived benchmark auth identity;
- a dedicated relay node pool, initially one `c1a.16x` (10 Gbps) or
  `c1a.32x` (20 Gbps), labeled and tainted for the relay;
- at least two dedicated generator nodes of the same or larger aggregate VPC
  bandwidth, with no GPU/network-operator workload that consumes most CPU;
- explicit node affinity and anti-affinity so relay and generators land where
  the scenario declares; and
- benchmark Jobs with a run ID/path prefix and a controller that always cleans
  up clients and credentials.

The existing `c1a.2x` k6 pool should not be reused unchanged. Either introduce
a video-specific generator pool or resize the staging-only benchmark pool and
exclude irrelevant heavyweight DaemonSets. Taints are necessary so unrelated
work does not consume the pool while a run is active.

## Harness changes before large tests

Build a controller plus a small load-agent image:

- Controller expands a versioned scenario manifest, creates distributed
  publisher/reader Jobs, waits for readiness, timestamps the measured interval,
  queries all metric backends, enforces stop conditions, and cleans up.
- Agent supports `publish-copy`, `read-copy`, `read-decode`, and WHEP probe
  roles, emits Prometheus metrics and a final JSON summary, and uses
  `ffmpeg -progress` rather than process liveness alone.
- Manifests define publisher/read locations independently: external, relay-node,
  another same-cell node, GPU-processor node, or remote cell.
- Reports include the fixture codec/resolution/FPS/bitrate/GOP/hash, requested
  and observed placement, actual bitrate, client resource use, relay and node
  resource use, all software versions, and counter deltas.
- Credentials are supplied only through Secrets/environment and are redacted
  from commands and reports.

Keep two reader modes. `read-copy` finds the network/relay knee without spending
CPU decoding every copy. `read-decode` verifies delivered FPS, corruption, and
time-to-first-frame on a statistically meaningful subset.

## Exact benchmark progression

### Phase 0: calibration

1. Run one publisher with no readers through the internal service.
2. Run one publisher and one copy reader through the internal service.
3. Add one decode reader and verify the fixture's expected FPS.
4. Repeat through the public RTSP LoadBalancer.
5. Repeat WHEP signaling and WebRTC media over the public TCP ICE port.
6. Confirm byte conservation across client, MediaMTX, pod, node, and Crusoe
   metrics within expected protocol overhead.

### Phase 1: isolate each same-cell edge

Run each edge independently before an end-to-end test:

1. relay process plus same-node pod networking;
2. direct pod-IP cross-node networking;
3. ClusterIP/Cilium cross-node networking;
4. public L4 LB ingress;
5. public L4 LB egress;
6. WHEP signaling through Traefik;
7. WebRTC media through the L4 LB; and
8. processor-node read and output-publish paths.

This prevents a single aggregate curve from hiding whether the LB, relay node,
Cilium overlay, processor node, or generator failed first.

### Phase 2: relay source/fan-out curves

Use encoded H.264 fixtures covering 720p, 1080p, and representative 4K inputs,
with 1/5/10/20 Mbps bitrates and short/long GOPs.

For each fixture:

1. Increase sources with zero readers: `1, 10, 25, 50, 100, 200, ...`.
2. Increase sources with one reader each.
3. Hold source count and sweep `1, 2, 4, 8` readers per source.
4. Add source WHEP previews.
5. Add watched annotated-output streams.
6. Run a connection/reconnection storm separately from steady throughput.

Run 10-15 minutes per ramp. Stop at the first agreed SLO or resource breach.
Run one hour at the last healthy point, then 24 hours near the proposed
certified operating limit.

### Phase 3: combined relay and processor tests

Use the same source fixtures while the workflow harness sweeps processor
concurrency. Compare:

- one relay reader per source;
- multiple workflows reading the same source;
- output publishing off;
- output publishing on but unwatched; and
- output publishing on and watched.

This is the only phase that produces a realistic whole-cell capacity envelope.
Relay-only and GPU-only maxima must not be multiplied together as if they were
independent.

### Phase 4: multi-cell and WAN

The staging South Central cluster already exists. It has `c1a.16x` nodes
(10 Gbps VPC) and `l40s-48gb.1x` nodes (17.5 Gbps VPC), but no video cell. Its
VPC/pod/service CIDRs are distinct from East, so the current infrastructure has
no cluster-local route between cells. Initial cross-cell media therefore needs
a public endpoint or an explicit tunnel/peering design.

The `video-proc` namespace exists in South Central but contains no relay or
processor workloads. East currently runs Cilium 1.17.16 while South Central
runs 1.19.1, so CNI behavior is not an exact A/B comparison unless the version
difference is accepted and recorded.

Compare:

1. East relay to East processor;
2. East public relay directly to each South Central processor;
3. East origin to one South Central read replica, then local fan-out;
4. GCP relay to Crusoe processor;
5. Crusoe relay to GCP processor; and
6. cloud relay to an on-prem/customer-style runner.

Sweep 20/50/100/200 ms RTT, 0/0.1/0.5/1/2% loss, and bandwidth caps near the
encoded rate using isolated `tc netem` agents. Compare the current RTSP/TCP path
with secure WAN candidates. South Central has a documented unresolved QUIC/UDP
egress flap, and it lacks the same Hubble Relay/metrics setup as East; correct
or explicitly account for those differences before treating a regional result
as a transport comparison.

For a remote cell with `N` processors reading one source at bitrate `B`, compare
direct fan-out (`N * B` WAN) with an origin/read-replica link (`B` WAN plus
`N * B` local). Record recovery behavior when the inter-cell link, origin, or
replica restarts.

## Provisional stop and certification gates

Stop a ramp when any of the following occurs:

- a publisher or reader disconnects or fails;
- ready path/reader counts differ from requested counts for more than two
  scrape intervals;
- RTSP/TCP loss or media input errors become non-zero, or decoded delivery
  falls below 95% of the fixture FPS;
- p95 startup/TTFR or continuity exceeds the product SLO (until set, use a
  two-times-baseline regression gate);
- relay CPU exceeds 80%, memory exceeds 80%, or sustained CFS throttling exceeds
  1% of CPU time;
- the relay node exceeds 70% of its documented VPC entitlement;
- node/pod drops or errors increase, TCP retransmission rate exceeds the agreed
  threshold, or Cilium/Linux conntrack pressure exceeds 50%; or
- LB bytes/flows plateau while clients request more load.

For each fixture/topology, record the knee and certify a lower operating point.
One-half to two-thirds of the measured knee is the initial headroom hypothesis;
the one-hour and 24-hour soaks determine the final value.

## Next implementation changes

1. Pin the normal staging relay to a stable relay-capable node class so ordinary
   measurements stop changing across restarts.
2. Add a dedicated performance-cell release and relay/generator node pools.
3. Expand the MediaMTX metric allowlist/dashboard with session bytes, RTP/RTCP
   packet and jitter counters, CPU throttling, pod drops/errors, relay-node NIC,
   TCP retransmits, Linux conntrack, and Cilium map/drop panels.
4. Add Crusoe metrics API collection for LB and VM VPC metrics.
5. Split the relay harness into distributed controller/agent components and add
   packet-copy plus decode-validation reader modes.
6. Add a staging-only, short-lived benchmark stream credential mechanism.
7. Run calibration and same-cell edge isolation before source/fan-out ramps.
8. Deploy the same benchmark cell contract to staging South Central only after
   East baselines are reproducible.
