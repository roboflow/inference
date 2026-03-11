#!/usr/bin/env python3
"""Lightweight API server for the RF-DETR few-shot grid search results explorer."""

import argparse
import json
import logging
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "results.db"
STATIC_DIR = Path(__file__).parent / "frontend"
IS_PHASE2 = False  # set by --db flag if phase 2 DB is detected
SYNC_MANAGER = None  # set by --remote flag


class RemoteSyncManager:
    """Background thread that syncs a remote SQLite DB to local via scp."""

    def __init__(self, remote_host, remote_db_path, local_db_path, interval=10):
        self.remote_host = remote_host  # e.g. "roboflow@100.94.130.94"
        self.remote_db_path = remote_db_path
        self.local_db_path = local_db_path
        self.interval = interval
        self._lock = threading.Lock()
        self._last_sync = None
        self._last_error = None
        self._sync_count = 0
        self._running = False
        self._thread = None

    def start(self):
        """Start background sync thread."""
        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logger.info("Sync thread started: %s every %ds", self.remote_host, self.interval)

    def stop(self):
        self._running = False

    def sync_now(self):
        """Trigger an immediate sync (can be called from API handler)."""
        return self._do_sync()

    def status(self):
        return {
            "enabled": True,
            "remote_host": self.remote_host,
            "last_sync": self._last_sync,
            "last_error": self._last_error,
            "sync_count": self._sync_count,
            "interval": self.interval,
        }

    def _sync_loop(self):
        while self._running:
            self._do_sync()
            time.sleep(self.interval)

    def _do_sync(self):
        """Run WAL checkpoint on remote, then scp the DB file."""
        with self._lock:
            try:
                # Step 1: WAL checkpoint on remote via Python (sqlite3 CLI not available)
                checkpoint_cmd = (
                    f"sshpass -p 'roboflow' ssh -o StrictHostKeyChecking=no "
                    f"-o ConnectTimeout=5 {self.remote_host} "
                    f"\"~/.pyenv/versions/inference-exp/bin/python -c \\\"import sqlite3,os; "
                    f"c=sqlite3.connect(os.path.expanduser('{self.remote_db_path}')); "
                    f"c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()\\\"\""
                )
                subprocess.run(
                    checkpoint_cmd, shell=True, capture_output=True,
                    timeout=15
                )

                # Step 2: scp the DB file (WAL should be empty after TRUNCATE checkpoint)
                scp_cmd = (
                    f"sshpass -p 'roboflow' scp -o StrictHostKeyChecking=no "
                    f"-o ConnectTimeout=5 "
                    f"{self.remote_host}:{self.remote_db_path} {self.local_db_path}"
                )
                result = subprocess.run(
                    scp_cmd, shell=True, capture_output=True,
                    timeout=30
                )

                if result.returncode == 0:
                    self._last_sync = datetime.now().isoformat()
                    self._last_error = None
                    self._sync_count += 1
                    return {"ok": True, "synced_at": self._last_sync}
                else:
                    err = result.stderr.decode().strip()[:200]
                    self._last_error = err
                    logger.warning("scp failed: %s", err)
                    return {"ok": False, "error": err}

            except subprocess.TimeoutExpired:
                self._last_error = "timeout"
                logger.warning("Sync timed out")
                return {"ok": False, "error": "timeout"}
            except Exception as e:
                self._last_error = str(e)[:200]
                logger.warning("Sync error: %s", e)
                return {"ok": False, "error": str(e)[:200]}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _detect_phase2(conn):
    """Check if DB has phase 2 columns."""
    cols = [c[1] for c in conn.execute("PRAGMA table_info(experiments)").fetchall()]
    return "lora_dropout" in cols


def _avg(values):
    """Average a list of values, skipping None."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed)
        else:
            super().do_GET()

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_api(self, parsed):
        qs = parse_qs(parsed.query)
        path = parsed.path
        conn = get_db()
        try:
            if path == "/api/summary":
                self._api_summary(conn)
            elif path == "/api/experiments":
                self._api_experiments(conn, qs)
            elif path == "/api/heatmap":
                self._api_heatmap(conn, qs)
            elif path == "/api/experiment":
                self._api_experiment_detail(conn, qs)
            elif path == "/api/experiment_group":
                self._api_experiment_group(conn, qs)
            elif path == "/api/compare":
                self._api_compare(conn, qs)
            elif path == "/api/progress":
                self._api_progress(conn)
            elif path == "/api/gpu_stats":
                self._api_gpu_stats(conn, qs)
            elif path == "/api/running":
                self._api_running(conn)
            elif path == "/api/sync":
                self._api_sync()
                return  # already sent response, skip finally close
            elif path == "/api/sync_status":
                self._api_sync_status()
                return
            else:
                self._json_response({"error": "unknown endpoint"}, 404)
        finally:
            conn.close()

    # ── Helpers ─────────────────────────────────────────────────

    def _enrich_experiments(self, conn, experiments):
        """Add per-class AP and conf_metrics to experiment rows.
        Returns (enriched_list, all_class_names)."""
        if not experiments:
            return [], []

        exp_ids = [e["id"] for e in experiments]
        placeholders = ",".join("?" * len(exp_ids))
        evals = conn.execute(
            f"SELECT * FROM eval_results WHERE experiment_id IN ({placeholders})",
            exp_ids,
        ).fetchall()

        # Group by experiment_id
        eval_by_exp = defaultdict(list)
        all_classes = set()
        for ev in evals:
            ev = dict(ev)
            eval_by_exp[ev["experiment_id"]].append(ev)
            if ev.get("per_class_ap_json"):
                all_classes.update(json.loads(ev["per_class_ap_json"]).keys())
        all_classes = sorted(all_classes)

        enriched = []
        for exp in experiments:
            row = dict(exp)
            evs = eval_by_exp.get(exp["id"], [])

            recalls, precisions, f1s = [], [], []
            class_aps = {cls: [] for cls in all_classes}

            for ev in evs:
                if ev.get("conf_metrics_json"):
                    cm = json.loads(ev["conf_metrics_json"])
                    for thresh in ("0.3",):
                        c = cm.get(thresh, {})
                        if c.get("recall") is not None:
                            recalls.append(c["recall"])
                        if c.get("precision") is not None:
                            precisions.append(c["precision"])
                        if c.get("f1") is not None:
                            f1s.append(c["f1"])

                if ev.get("per_class_ap_json"):
                    pcap = json.loads(ev["per_class_ap_json"])
                    for cls in all_classes:
                        if cls in pcap:
                            ap50 = pcap[cls].get("0.5")
                            if ap50 is not None:
                                class_aps[cls].append(ap50)

            row["recall_03"] = _avg(recalls)
            row["precision_03"] = _avg(precisions)
            row["f1_03"] = _avg(f1s)
            row["per_class_ap50"] = {
                cls: round(_avg(vs), 4) if vs else None
                for cls, vs in class_aps.items()
            }
            enriched.append(row)

        return enriched, all_classes

    def _aggregate_experiments(self, experiments, all_classes, require_all_sets=True):
        """Group enriched experiments by hyperparams, averaging metrics across sets.

        If require_all_sets is True, only include configs where all training sets
        are present — incomplete configs (only one set done) are excluded.
        """
        # Count total distinct sets in the data
        all_sets = set(e["train_image_set"] for e in experiments) if require_all_sets else set()
        n_total_sets = len(all_sets) if all_sets else 1

        groups = defaultdict(list)
        for exp in experiments:
            key = (exp["lora_rank"], exp["num_epochs"],
                   exp["learning_rate"], exp["num_train_images"])
            # Phase 2: extend key with additional params
            if IS_PHASE2:
                key = key + (
                    exp.get("lora_dropout"),
                    exp.get("augmentation_level"),
                    exp.get("weight_decay"),
                    exp.get("alpha_ratio"),
                )
            groups[key].append(exp)

        aggregated = []
        for key, exps in groups.items():
            agg = {
                "lora_rank": key[0],
                "num_epochs": key[1],
                "learning_rate": key[2],
                "num_train_images": key[3],
                "mAP_50": _avg([e["mAP_50"] for e in exps]),
                "mAP_50_95": _avg([e["mAP_50_95"] for e in exps]),
                "train_time_seconds": _avg([e["train_time_seconds"] for e in exps]),
                "time_per_epoch_ms": _avg([e["time_per_epoch_ms"] for e in exps]),
                "final_loss": _avg([e["final_loss"] for e in exps]),
                "recall_03": _avg([e["recall_03"] for e in exps]),
                "precision_03": _avg([e["precision_03"] for e in exps]),
                "f1_03": _avg([e["f1_03"] for e in exps]),
                "n_sets": len(exps),
                "sets": ",".join(sorted(set(e["train_image_set"] for e in exps))),
                "experiment_ids": ",".join(str(e["id"]) for e in exps),
                "per_class_ap50": {},
            }
            if IS_PHASE2:
                agg["lora_dropout"] = key[4]
                agg["augmentation_level"] = key[5]
                agg["weight_decay"] = key[6]
                agg["alpha_ratio"] = key[7]
            for cls in all_classes:
                vals = [e["per_class_ap50"].get(cls) for e in exps
                        if e["per_class_ap50"].get(cls) is not None]
                agg["per_class_ap50"][cls] = (
                    round(sum(vals) / len(vals), 4) if vals else None
                )
            aggregated.append(agg)

        # Filter out incomplete configs (not all training sets done)
        if require_all_sets and n_total_sets > 1:
            aggregated = [a for a in aggregated if a["n_sets"] >= n_total_sets]

        aggregated.sort(key=lambda x: x["mAP_50"] or 0, reverse=True)
        return aggregated

    # ── API Endpoints ──────────────────────────────────────────

    def _api_summary(self, conn):
        meta_row = conn.execute(
            "SELECT value FROM grid_meta WHERE key='total_experiments'"
        ).fetchone()
        grid_total = int(meta_row[0]) if meta_row else None

        completed = conn.execute(
            "SELECT count(*) FROM experiments WHERE status='completed'"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT count(*) FROM experiments WHERE status='failed'"
        ).fetchone()[0]
        running = conn.execute(
            "SELECT count(*) FROM experiments WHERE status='running'"
        ).fetchone()[0]

        eta_info = self._compute_eta(conn, grid_total, completed)

        # Elapsed time — compute from actual experiment timestamps (avoids timezone issues
        # between remote server clock and local dashboard server clock)
        elapsed = None
        is_running = running > 0
        first_exp = conn.execute(
            "SELECT MIN(timestamp) FROM experiments WHERE status='completed'"
        ).fetchone()
        last_exp = conn.execute(
            "SELECT MAX(timestamp) FROM experiments WHERE status='completed'"
        ).fetchone()
        if first_exp and first_exp[0] and last_exp and last_exp[0]:
            try:
                first_dt = datetime.fromisoformat(first_exp[0])
                last_dt = datetime.fromisoformat(last_exp[0])
                elapsed = (last_dt - first_dt).total_seconds()
                # If still running, add time since last completed experiment
                # (estimated from avg experiment time)
                if is_running and completed > 0:
                    avg_row = conn.execute(
                        "SELECT AVG(train_time_seconds) FROM experiments WHERE status='completed'"
                    ).fetchone()
                    if avg_row and avg_row[0]:
                        elapsed += avg_row[0]  # approximate
            except Exception:
                elapsed = None

        # Phase 2 group-by includes extra columns
        if IS_PHASE2:
            group_cols = "lora_rank, num_epochs, learning_rate, num_train_images, lora_dropout, augmentation_level, weight_decay, alpha_ratio"
        else:
            group_cols = "lora_rank, num_epochs, learning_rate, num_train_images"

        # Count how many distinct training sets exist in the grid
        n_total_sets = conn.execute(
            "SELECT COUNT(DISTINCT train_image_set) FROM experiments"
        ).fetchone()[0] or 1

        best = conn.execute(f"""
            SELECT {group_cols},
                   AVG(mAP_50) as mAP_50, AVG(mAP_50_95) as mAP_50_95,
                   AVG(train_time_seconds) as train_time_seconds,
                   COUNT(*) as n_sets,
                   GROUP_CONCAT(DISTINCT train_image_set) as sets
            FROM experiments WHERE status = 'completed'
            GROUP BY {group_cols}
            HAVING COUNT(DISTINCT train_image_set) = {n_total_sets}
            ORDER BY AVG(mAP_50) DESC LIMIT 10
        """).fetchall()

        self._json_response({
            "grid_total": grid_total,
            "completed": completed, "failed": failed, "running": running,
            "best_runs": [dict(r) for r in best],
            "elapsed_seconds": elapsed,
            "is_running": is_running,
            "is_phase2": IS_PHASE2,
            **eta_info,
        })

    def _compute_eta(self, conn, grid_total, completed):
        if not grid_total or completed == 0:
            return {"eta_seconds": None, "avg_time_per_experiment": None,
                    "pct_complete": 0}

        pct = round(100 * completed / grid_total, 1)

        # Simple approach that works for both phases: average time × remaining
        row = conn.execute("""
            SELECT AVG(train_time_seconds) as avg_time,
                   COUNT(*) as cnt,
                   SUM(train_time_seconds) as total_time
            FROM experiments WHERE status='completed' AND train_time_seconds IS NOT NULL
        """).fetchone()

        if not row or not row["avg_time"]:
            return {"eta_seconds": None, "avg_time_per_experiment": None,
                    "pct_complete": pct}

        avg_time = row["avg_time"]
        eval_overhead = 2.0  # eval + DB overhead per experiment
        remaining_count = grid_total - completed
        running = conn.execute(
            "SELECT count(*) FROM experiments WHERE status='running'"
        ).fetchone()[0]
        remaining_count = max(0, remaining_count - running)

        # Use recent experiments for more accurate estimate (last 20)
        recent = conn.execute("""
            SELECT AVG(train_time_seconds) as avg_time
            FROM (SELECT train_time_seconds FROM experiments
                  WHERE status='completed' AND train_time_seconds IS NOT NULL
                  ORDER BY timestamp DESC LIMIT 20)
        """).fetchone()
        recent_avg = recent["avg_time"] if recent and recent["avg_time"] else avg_time

        # Blend recent and overall (weight recent more)
        blended_avg = recent_avg * 0.7 + avg_time * 0.3

        # Account for number of workers
        workers_row = conn.execute(
            "SELECT value FROM grid_meta WHERE key='num_workers'"
        ).fetchone()
        num_workers = int(workers_row[0]) if workers_row else max(1, running)
        if num_workers < 1:
            num_workers = 1

        eta = (remaining_count * (blended_avg + eval_overhead)) / num_workers

        return {
            "eta_seconds": round(eta, 0),
            "avg_time_per_experiment": round(avg_time, 1),
            "pct_complete": pct,
            "remaining_count": remaining_count,
        }

    def _api_experiments(self, conn, qs):
        group = qs.get("group", ["avg"])[0]
        train_set = qs.get("train_set", [None])[0]

        where = ["status = 'completed'"]
        if train_set:
            where.append(f"train_image_set = '{train_set}'")

        experiments = conn.execute(f"""
            SELECT * FROM experiments WHERE {' AND '.join(where)}
            ORDER BY mAP_50 DESC
        """).fetchall()

        enriched, all_classes = self._enrich_experiments(conn, experiments)

        if group == "individual":
            self._json_response({
                "rows": enriched, "classes": all_classes, "view": "individual",
            })
        else:
            aggregated = self._aggregate_experiments(enriched, all_classes)
            self._json_response({
                "rows": aggregated, "classes": all_classes, "view": "avg",
            })

    def _api_heatmap(self, conn, qs):
        metric = qs.get("metric", ["mAP_50"])[0]
        n_train = qs.get("n_train", [None])[0]
        train_set = qs.get("train_set", [None])[0]
        col = {"mAP_50": "mAP_50", "mAP_50_95": "mAP_50_95"}[metric]

        where = ["status = 'completed'"]
        if n_train:
            where.append(f"num_train_images = {int(n_train)}")
        if train_set:
            where.append(f"train_image_set = '{train_set}'")
        where_sql = " AND ".join(where)

        if IS_PHASE2:
            group_cols = "lora_rank, num_epochs, learning_rate, num_train_images, lora_dropout, augmentation_level, weight_decay, alpha_ratio"
        else:
            group_cols = "lora_rank, num_epochs, learning_rate, num_train_images"

        # When averaging across sets (no specific set filter), only include
        # configs where all training sets are completed
        having_sql = ""
        if not train_set:
            n_total_sets = conn.execute(
                "SELECT COUNT(DISTINCT train_image_set) FROM experiments"
            ).fetchone()[0] or 1
            having_sql = f"HAVING COUNT(DISTINCT train_image_set) = {n_total_sets}"

        rows = conn.execute(f"""
            SELECT {group_cols},
                   AVG({col}) as metric_val,
                   AVG(train_time_seconds) as train_time_seconds,
                   AVG(time_per_epoch_ms) as time_per_epoch_ms
            FROM experiments WHERE {where_sql}
            GROUP BY {group_cols}
            {having_sql}
        """).fetchall()
        self._json_response([dict(r) for r in rows])

    def _api_experiment_detail(self, conn, qs):
        exp_id = int(qs.get("id", [0])[0])
        exp = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
        if not exp:
            return self._json_response({"error": "not found"}, 404)

        evals = conn.execute(
            "SELECT * FROM eval_results WHERE experiment_id = ?", (exp_id,)
        ).fetchall()
        train_imgs = conn.execute(
            "SELECT * FROM train_images WHERE experiment_id = ?", (exp_id,)
        ).fetchall()

        self._json_response({
            "experiment": dict(exp),
            "eval_results": [dict(r) for r in evals],
            "train_images": [dict(r) for r in train_imgs],
        })

    def _api_experiment_group(self, conn, qs):
        """Get detail for an aggregated group — both sets with eval results."""
        rank = int(qs.get("rank", [0])[0])
        epochs = int(qs.get("epochs", [0])[0])
        lr = float(qs.get("lr", [0])[0])
        n_train = int(qs.get("n_train", [0])[0])

        params = [rank, epochs, lr, n_train]
        extra_where = ""
        if IS_PHASE2:
            dropout = float(qs.get("dropout", [0])[0])
            aug_level = int(qs.get("aug_level", [0])[0])
            wd = float(qs.get("wd", [0])[0])
            alpha_ratio = int(qs.get("alpha_ratio", [0])[0])
            extra_where = " AND lora_dropout = ? AND augmentation_level = ? AND weight_decay = ? AND alpha_ratio = ?"
            params.extend([dropout, aug_level, wd, alpha_ratio])

        experiments = conn.execute(f"""
            SELECT * FROM experiments
            WHERE lora_rank = ? AND num_epochs = ? AND learning_rate = ?
              AND num_train_images = ? {extra_where} AND status = 'completed'
            ORDER BY train_image_set
        """, params).fetchall()

        result = []
        for exp in experiments:
            exp_d = dict(exp)
            evals = conn.execute(
                "SELECT * FROM eval_results WHERE experiment_id = ?",
                (exp_d["id"],),
            ).fetchall()
            train_imgs = conn.execute(
                "SELECT * FROM train_images WHERE experiment_id = ?",
                (exp_d["id"],),
            ).fetchall()
            result.append({
                "experiment": exp_d,
                "eval_results": [dict(r) for r in evals],
                "train_images": [dict(r) for r in train_imgs],
            })

        self._json_response(result)

    def _api_compare(self, conn, qs):
        vary = qs.get("vary", ["lora_rank"])[0]
        metric = qs.get("metric", ["mAP_50"])[0]
        col = {"mAP_50": "mAP_50", "mAP_50_95": "mAP_50_95"}.get(metric, "mAP_50")

        rows = conn.execute(f"""
            SELECT {vary},
                   AVG({col}) as avg_metric, MIN({col}) as min_metric,
                   MAX({col}) as max_metric,
                   AVG(train_time_seconds) as avg_train_time,
                   COUNT(*) as n_experiments
            FROM experiments WHERE status = 'completed'
            GROUP BY {vary} ORDER BY {vary}
        """).fetchall()
        self._json_response([dict(r) for r in rows])

    def _api_progress(self, conn):
        extra = ", lora_dropout, augmentation_level, weight_decay, alpha_ratio" if IS_PHASE2 else ""
        rows = conn.execute(f"""
            SELECT timestamp, status, run_id, train_time_seconds,
                   lora_rank, num_epochs, learning_rate, num_train_images,
                   mAP_50, mAP_50_95{extra}
            FROM experiments ORDER BY timestamp
        """).fetchall()
        self._json_response([dict(r) for r in rows])

    def _api_running(self, conn):
        """Return currently running experiments with duration."""
        rows = conn.execute("""
            SELECT id, run_id, timestamp, lora_rank, num_epochs, learning_rate,
                   num_train_images, train_image_set,
                   lora_dropout, augmentation_level, weight_decay, alpha_ratio
            FROM experiments WHERE status = 'running'
            ORDER BY timestamp
        """).fetchall()

        running = []
        for r in rows:
            rd = dict(r)
            # Compute how long it's been running
            try:
                started = datetime.fromisoformat(rd["timestamp"])
                rd["running_seconds"] = (datetime.now() - started).total_seconds()
            except Exception:
                rd["running_seconds"] = None
            running.append(rd)

        self._json_response({"running": running, "count": len(running)})

    def _api_gpu_stats(self, conn, qs):
        """Return GPU utilization history for sparkline display."""
        # Check if gpu_stats table exists
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "gpu_stats" not in tables:
            return self._json_response({"points": [], "latest": None})

        limit = int(qs.get("limit", ["200"])[0])
        since = qs.get("since", [None])[0]

        if since:
            rows = conn.execute(
                "SELECT * FROM gpu_stats WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?",
                (since, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM gpu_stats ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()

        points = [dict(r) for r in reversed(rows)]
        latest = points[-1] if points else None

        self._json_response({"points": points, "latest": latest})

    def _api_sync(self):
        """Trigger an immediate DB sync from remote."""
        if not SYNC_MANAGER:
            return self._json_response({"ok": False, "error": "No remote sync configured. Use --remote flag."})
        result = SYNC_MANAGER.sync_now()
        self._json_response(result)

    def _api_sync_status(self):
        """Return current sync status."""
        if not SYNC_MANAGER:
            return self._json_response({"enabled": False})
        self._json_response(SYNC_MANAGER.status())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results.db",
                        help="Database file (e.g. results.db or results_phase2.db)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--remote", type=str, default=None,
                        help="Remote host for DB sync, e.g. roboflow@100.94.130.94")
    parser.add_argument("--remote-db", type=str, default=None,
                        help="Path to DB on remote host (default: auto-detect from --db)")
    parser.add_argument("--sync-interval", type=int, default=10,
                        help="Seconds between background syncs (default: 10)")
    args = parser.parse_args()

    DB_PATH = Path(__file__).parent / args.db

    # Start remote sync if configured
    if args.remote:
        remote_db = args.remote_db
        if not remote_db:
            # Default: ~/inference/experiments/rfdetr_few_shot/<db_name>
            remote_db = f"~/inference/experiments/rfdetr_few_shot/{args.db}"
        SYNC_MANAGER = RemoteSyncManager(
            remote_host=args.remote,
            remote_db_path=remote_db,
            local_db_path=str(DB_PATH),
            interval=args.sync_interval,
        )
        # Do an initial sync before starting the server
        print(f"🔄 Initial sync from {args.remote}:{remote_db} ...")
        result = SYNC_MANAGER.sync_now()
        if result.get("ok"):
            print(f"✅ Initial sync complete")
        else:
            print(f"⚠️  Initial sync failed: {result.get('error')} — will retry in background")
        SYNC_MANAGER.start()

    # Auto-detect phase 2
    if DB_PATH.exists():
        _conn = sqlite3.connect(str(DB_PATH))
        IS_PHASE2 = _detect_phase2(_conn)
        _conn.close()
        if IS_PHASE2:
            print(f"📊 Phase 2 DB detected: {DB_PATH}")

    port = args.port
    sync_info = f" · syncing from {args.remote} every {args.sync_interval}s" if args.remote else ""
    print(f"🚀 Results explorer: http://localhost:{port}{sync_info}")
    HTTPServer(("", port), Handler).serve_forever()
