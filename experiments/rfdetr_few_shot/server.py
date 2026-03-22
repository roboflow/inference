#!/usr/bin/env python3
"""Lightweight API server for the RF-DETR few-shot grid search results explorer."""

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "results.db"
COMPARE_DB_PATH = None  # optional second DB for LoRA vs FT comparison
STATIC_DIR = Path(__file__).parent / "frontend"
IS_PHASE2 = False  # set by --db flag if phase 2 DB is detected
IS_BENCHMARK = False  # set by --db flag if benchmark DB is detected
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


_DB_LOCAL = threading.local()
_DB_INIT_LOCK = threading.Lock()
_DB_INDEXES_CREATED = False

def get_db():
    """Return a per-thread read-only DB connection (WAL mode allows concurrent reads)."""
    global _DB_INDEXES_CREATED, DB_COLUMNS, IS_BENCHMARK, IS_PHASE2
    conn = getattr(_DB_LOCAL, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _DB_LOCAL.conn = conn
        # Create indexes once (first thread to connect)
        if not _DB_INDEXES_CREATED:
            with _DB_INIT_LOCK:
                if not _DB_INDEXES_CREATED:
                    try:
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_exp_id ON eval_results(experiment_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_name)")
                    except Exception:
                        pass
                    _DB_INDEXES_CREATED = True
    # Always refresh DB_COLUMNS — schema can change at runtime (e.g. reeval adds lora columns)
    if DB_PATH.exists():
        try:
            new_cols = {c[1] for c in conn.execute("PRAGMA table_info(experiments)").fetchall()}
            if new_cols and new_cols != DB_COLUMNS:
                DB_COLUMNS = new_cols
                IS_BENCHMARK = _detect_benchmark(conn)
                IS_PHASE2 = _detect_phase2(conn)
        except Exception:
            pass
    return conn


_COMPARE_DB_LOCAL = threading.local()

def get_compare_db():
    """Return a per-thread read-only connection to the comparison DB, or None."""
    if COMPARE_DB_PATH is None or not COMPARE_DB_PATH.exists():
        return None
    conn = getattr(_COMPARE_DB_LOCAL, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(str(COMPARE_DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _COMPARE_DB_LOCAL.conn = conn
    return conn


def _detect_phase2(conn):
    """Check if DB has phase 2 columns."""
    cols = [c[1] for c in conn.execute("PRAGMA table_info(experiments)").fetchall()]
    return "lora_dropout" in cols


def _detect_benchmark(conn):
    """Check if DB has benchmark columns (dataset_name)."""
    cols = [c[1] for c in conn.execute("PRAGMA table_info(experiments)").fetchall()]
    return "dataset_name" in cols


def _avg(values):
    """Average a list of values, skipping None."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            pass  # client disconnected during init/handle — harmless

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            self.close_connection = True

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api(parsed)
            elif parsed.path.startswith("/viz/") or parsed.path.startswith("/viz_champion/"):
                self._serve_viz_file(parsed.path)
            else:
                super().do_GET()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass  # client disconnected, ignore
        except Exception:
            logger.exception("Unhandled error in do_GET")

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
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
            elif path == "/api/datasets":
                self._api_datasets(conn)
            elif path == "/api/benchmark_heatmap":
                self._api_benchmark_heatmap(conn, qs)
            elif path == "/api/sync":
                self._api_sync()
                return  # already sent response, skip finally close
            elif path == "/api/sync_status":
                self._api_sync_status()
                return
            elif path == "/api/viz_champion":
                self._api_viz_champion()
                return
            elif path == "/api/config_detail":
                self._api_config_detail(conn, qs)
            elif path == "/api/ft_comparison":
                self._api_ft_comparison(conn, qs)
            else:
                self._json_response({"error": "unknown endpoint"}, 404)
        except Exception:
            logger.exception("API error on %s", path)
            self._json_response({"error": "internal server error"}, 500)

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
        """Group enriched experiments by hyperparams, averaging metrics across sets/datasets.

        If require_all_sets is True, only include configs where all training sets
        (or datasets in benchmark mode) are present.
        """
        # Count total distinct sets in the data
        if IS_BENCHMARK:
            all_sets = set(e["dataset_name"] for e in experiments if e.get("dataset_name")) if require_all_sets else set()
        else:
            all_sets = set(e["train_image_set"] for e in experiments) if require_all_sets else set()
        n_total_sets = len(all_sets) if all_sets else 1

        # Build grouping key fields based on DB type
        if IS_BENCHMARK:
            bm_key_fields = ["lora_rank", "num_epochs", "learning_rate",
                             "lora_dropout", "augmentation_level", "weight_decay", "alpha_ratio"]
            for col in ["group_detr", "batch_size", "lora_targets", "copy_paste", "mosaic", "warmup", "multi_scale"]:
                if col in DB_COLUMNS:
                    bm_key_fields.append(col)

        groups = defaultdict(list)
        for exp in experiments:
            if IS_BENCHMARK:
                key = tuple(exp.get(f) for f in bm_key_fields)
            elif IS_PHASE2:
                key = (exp["lora_rank"], exp["num_epochs"],
                       exp["learning_rate"], exp["num_train_images"],
                       exp.get("lora_dropout"),
                       exp.get("augmentation_level"),
                       exp.get("weight_decay"),
                       exp.get("alpha_ratio"),
                       )
            else:
                key = (exp["lora_rank"], exp["num_epochs"],
                       exp["learning_rate"], exp["num_train_images"])
            groups[key].append(exp)

        aggregated = []
        for key, exps in groups.items():
            agg = {
                "mAP_50": _avg([e["mAP_50"] for e in exps]),
                "mAP_50_95": _avg([e["mAP_50_95"] for e in exps]),
                "train_time_seconds": _avg([e["train_time_seconds"] for e in exps]),
                "time_per_epoch_ms": _avg([e["time_per_epoch_ms"] for e in exps]),
                "final_loss": _avg([e["final_loss"] for e in exps]),
                "recall_03": _avg([e.get("recall_03") for e in exps]),
                "precision_03": _avg([e.get("precision_03") for e in exps]),
                "f1_03": _avg([e.get("f1_03") for e in exps]),
                "n_sets": len(exps),
                "experiment_ids": ",".join(str(e["id"]) for e in exps),
                "per_class_ap50": {},
            }

            if IS_BENCHMARK:
                for i, field in enumerate(bm_key_fields):
                    agg[field] = key[i]
                agg["n_datasets"] = len(set(e.get("dataset_name") for e in exps))
                agg["datasets"] = ",".join(sorted(set(e.get("dataset_name", "") for e in exps)))
                agg["num_train_images"] = exps[0].get("num_train_images")
                agg["sets"] = agg["datasets"]
            elif IS_PHASE2:
                agg["lora_rank"] = key[0]
                agg["num_epochs"] = key[1]
                agg["learning_rate"] = key[2]
                agg["num_train_images"] = key[3]
                agg["lora_dropout"] = key[4]
                agg["augmentation_level"] = key[5]
                agg["weight_decay"] = key[6]
                agg["alpha_ratio"] = key[7]
                agg["sets"] = ",".join(sorted(set(e["train_image_set"] for e in exps)))
            else:
                agg["lora_rank"] = key[0]
                agg["num_epochs"] = key[1]
                agg["learning_rate"] = key[2]
                agg["num_train_images"] = key[3]
                agg["sets"] = ",".join(sorted(set(e["train_image_set"] for e in exps)))
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

        # Include any experiments added after grid_meta was set (e.g. 1000-epoch runs)
        actual_total = conn.execute("SELECT count(*) FROM experiments").fetchone()[0]
        if grid_total is None:
            grid_total = actual_total
        else:
            grid_total = max(grid_total, actual_total)

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

        # Phase 2 / benchmark group-by includes extra columns
        # Build group_cols from columns that actually exist in the DB
        best = []
        n_total_sets = 1
        set_col = "dataset_name" if "dataset_name" in DB_COLUMNS else "train_image_set"
        try:
            if IS_BENCHMARK:
                base_bm_cols = "lora_rank, num_epochs, learning_rate, lora_dropout, augmentation_level, weight_decay, alpha_ratio"
                if "group_detr" in DB_COLUMNS:
                    group_cols = base_bm_cols + ", group_detr"
                else:
                    # Phase 3+ benchmark without group_detr
                    extras = []
                    for col in ["batch_size", "lora_targets", "copy_paste", "mosaic", "warmup", "multi_scale"]:
                        if col in DB_COLUMNS:
                            extras.append(col)
                    group_cols = base_bm_cols + (", " + ", ".join(extras) if extras else "")
            elif IS_PHASE2:
                group_cols = "lora_rank, num_epochs, learning_rate, num_train_images, lora_dropout, augmentation_level, weight_decay, alpha_ratio"
            elif "lora_rank" in DB_COLUMNS:
                group_cols = "lora_rank, num_epochs, learning_rate, num_train_images"
            else:
                # Finetune baseline or other non-LoRA DB — group by model_variant
                group_cols = "model_variant, num_epochs, learning_rate, batch_size"

            # Count how many distinct training sets / datasets exist in the grid
            if "dataset_name" in DB_COLUMNS:
                n_total_sets = conn.execute(
                    "SELECT COUNT(DISTINCT dataset_name) FROM experiments"
                ).fetchone()[0] or 1
                set_col = "dataset_name"
            elif "train_image_set" in DB_COLUMNS:
                n_total_sets = conn.execute(
                    "SELECT COUNT(DISTINCT train_image_set) FROM experiments"
                ).fetchone()[0] or 1
                set_col = "train_image_set"

            best = conn.execute(f"""
                SELECT {group_cols},
                       AVG(mAP_50) as mAP_50, AVG(mAP_50_95) as mAP_50_95,
                       AVG(train_time_seconds) as train_time_seconds,
                       COUNT(*) as n_sets,
                       GROUP_CONCAT(DISTINCT {set_col}) as sets
                FROM experiments WHERE status = 'completed'
                GROUP BY {group_cols}
                HAVING COUNT(DISTINCT {set_col}) = {n_total_sets}
                ORDER BY AVG(mAP_50) DESC LIMIT 100
            """).fetchall()
        except Exception:
            best = []

        # Count datasets in benchmark mode
        n_datasets = 0
        if IS_BENCHMARK:
            n_datasets = conn.execute(
                "SELECT COUNT(DISTINCT dataset_name) FROM experiments WHERE status='completed'"
            ).fetchone()[0] or 0

        self._json_response({
            "grid_total": grid_total,
            "completed": completed, "failed": failed, "running": running,
            "best_runs": [dict(r) for r in best],
            "elapsed_seconds": elapsed,
            "is_running": is_running,
            "is_phase2": IS_PHASE2,
            "is_benchmark": IS_BENCHMARK,
            "n_datasets": n_datasets,
            **eta_info,
        })

    def _compute_eta(self, conn, grid_total, completed):
        if not grid_total or completed == 0:
            return {"eta_seconds": None, "avg_time_per_experiment": None,
                    "pct_complete": 0}

        pct = round(100 * completed / grid_total, 1)

        # Method-aware ETA: compute avg time per (method, model_variant) bucket,
        # then estimate remaining based on what's left to run in each bucket.
        has_method = "method" in DB_COLUMNS
        has_variant = "model_variant" in DB_COLUMNS

        if has_method and has_variant:
            # Get avg time per bucket from completed experiments
            bucket_avgs = {}
            rows = conn.execute("""
                SELECT method, model_variant,
                       AVG(train_time_seconds) as avg_time, COUNT(*) as cnt
                FROM experiments
                WHERE status='completed' AND train_time_seconds IS NOT NULL
                GROUP BY method, model_variant
            """).fetchall()
            for r in rows:
                bucket_avgs[(r["method"], r["model_variant"])] = r["avg_time"]

            # Count remaining per bucket (pending + running)
            remaining_rows = conn.execute("""
                SELECT method, model_variant, COUNT(*) as cnt
                FROM experiments WHERE status IN ('pending', 'running')
                GROUP BY method, model_variant
            """).fetchall()

            # Also count experiments not yet inserted (total - actual)
            actual_total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            unqueued = max(0, grid_total - actual_total)

            # Estimate remaining time
            overall_avg = sum(bucket_avgs.values()) / len(bucket_avgs) if bucket_avgs else 60
            eta = 0
            for r in remaining_rows:
                avg = bucket_avgs.get((r["method"], r["model_variant"]), overall_avg)
                eta += r["cnt"] * avg

            # For unqueued experiments, use overall average
            eta += unqueued * overall_avg

            avg_time = overall_avg
        else:
            # Fallback: simple average
            row = conn.execute("""
                SELECT AVG(train_time_seconds) as avg_time
                FROM experiments WHERE status='completed' AND train_time_seconds IS NOT NULL
            """).fetchone()
            avg_time = row["avg_time"] if row and row["avg_time"] else 60
            running = conn.execute(
                "SELECT count(*) FROM experiments WHERE status='running'"
            ).fetchone()[0]
            remaining_count = max(0, grid_total - completed - running)
            eta = remaining_count * avg_time

        return {
            "eta_seconds": round(eta, 0),
            "avg_time_per_experiment": round(avg_time, 1),
            "pct_complete": pct,
        }

    # Cache for enriched experiments: key -> (count, enriched, classes)
    _enrich_cache = {}

    def _api_experiments(self, conn, qs):
        group = qs.get("group", ["avg"])[0]
        train_set = qs.get("train_set", [None])[0]
        dataset = qs.get("dataset", [None])[0]

        where = ["status = 'completed'"]
        if train_set:
            where.append(f"train_image_set = '{train_set}'")
        if dataset and IS_BENCHMARK:
            where.append(f"dataset_name = '{dataset}'")

        where_sql = ' AND '.join(where)

        # For "avg" view in benchmark mode, use fast SQL aggregation (no eval_results)
        if group != "individual" and IS_BENCHMARK:
            return self._api_experiments_fast_avg(conn, where_sql, dataset)

        cache_key = where_sql

        # Check if cache is still valid (count hasn't changed)
        current_count = conn.execute(
            f"SELECT COUNT(*) FROM experiments WHERE {where_sql}"
        ).fetchone()[0]

        cached = type(self)._enrich_cache.get(cache_key)
        if cached and cached[0] == current_count:
            enriched, all_classes = cached[1], cached[2]
        else:
            # Exclude loss_history from bulk query to reduce payload size
            sel_cols = [c for c in DB_COLUMNS if c != "loss_history"]
            experiments = conn.execute(f"""
                SELECT {', '.join(sorted(sel_cols))} FROM experiments WHERE {where_sql}
                ORDER BY mAP_50 DESC
            """).fetchall()

            enriched, all_classes = self._enrich_experiments(conn, experiments)
            type(self)._enrich_cache[cache_key] = (current_count, enriched, all_classes)

        if group == "individual":
            self._json_response({
                "rows": enriched, "classes": all_classes, "view": "individual",
            })
        else:
            aggregated = self._aggregate_experiments(enriched, all_classes)
            self._json_response({
                "rows": aggregated, "classes": all_classes, "view": "avg",
            })

    def _api_experiments_fast_avg(self, conn, where_sql, dataset_filter):
        """Fast SQL-only aggregation for benchmark avg view — skips eval_results."""
        # Build grouping columns
        group_cols = ["lora_rank", "num_epochs", "learning_rate",
                      "lora_dropout", "augmentation_level", "weight_decay", "alpha_ratio"]
        for col in ["batch_size", "lora_targets", "copy_paste", "mosaic", "warmup", "multi_scale"]:
            if col in DB_COLUMNS:
                group_cols.append(col)
        group_sql = ", ".join(group_cols)

        # Count total datasets for completeness filter
        if not dataset_filter:
            n_total = conn.execute(
                "SELECT COUNT(DISTINCT dataset_name) FROM experiments WHERE status='completed'"
            ).fetchone()[0] or 1
            having = f"HAVING COUNT(DISTINCT dataset_name) = {n_total}"
        else:
            having = ""

        rows = conn.execute(f"""
            SELECT {group_sql},
                   AVG(mAP_50) as mAP_50,
                   AVG(mAP_50_95) as mAP_50_95,
                   AVG(train_time_seconds) as train_time_seconds,
                   AVG(time_per_epoch_ms) as time_per_epoch_ms,
                   AVG(final_loss) as final_loss,
                   COUNT(*) as n_sets,
                   COUNT(DISTINCT dataset_name) as n_datasets,
                   GROUP_CONCAT(DISTINCT dataset_name) as datasets,
                   MIN(num_train_images) as num_train_images,
                   MAX(id) as max_id
            FROM experiments WHERE {where_sql}
            GROUP BY {group_sql}
            {having}
            ORDER BY mAP_50 DESC
        """).fetchall()

        aggregated = []
        for r in rows:
            row = dict(r)
            row["sets"] = row.pop("datasets", "")
            row["per_class_ap50"] = {}
            row["recall_03"] = None
            row["precision_03"] = None
            row["f1_03"] = None
            aggregated.append(row)

        self._json_response({
            "rows": aggregated, "classes": [], "view": "avg",
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

        dataset = qs.get("dataset", [None])[0]
        if dataset and IS_BENCHMARK:
            where.append(f"dataset_name = '{dataset}'")

        if IS_BENCHMARK:
            base_bm_cols2 = "lora_rank, num_epochs, learning_rate, lora_dropout, augmentation_level, weight_decay, alpha_ratio"
            if "group_detr" in DB_COLUMNS:
                group_cols = base_bm_cols2 + ", group_detr"
            else:
                extras = []
                for col in ["batch_size", "lora_targets", "copy_paste", "mosaic", "warmup", "multi_scale"]:
                    if col in DB_COLUMNS:
                        extras.append(col)
                group_cols = base_bm_cols2 + (", " + ", ".join(extras) if extras else "")
        elif IS_PHASE2:
            group_cols = "lora_rank, num_epochs, learning_rate, num_train_images, lora_dropout, augmentation_level, weight_decay, alpha_ratio"
        else:
            group_cols = "lora_rank, num_epochs, learning_rate, num_train_images"

        # When averaging across sets (no specific set filter), only include
        # configs where all training sets/datasets are completed
        having_sql = ""
        if not train_set and not dataset:
            if IS_BENCHMARK:
                n_total_sets = conn.execute(
                    "SELECT COUNT(DISTINCT dataset_name) FROM experiments"
                ).fetchone()[0] or 1
                having_sql = f"HAVING COUNT(DISTINCT dataset_name) = {n_total_sets}"
            else:
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
        set_col = "dataset_name" if IS_BENCHMARK else "train_image_set"
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
            ORDER BY {set_col}
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
        base = ["timestamp", "status", "run_id", "train_time_seconds",
                "lora_rank", "num_epochs", "learning_rate", "num_train_images",
                "mAP_50", "mAP_50_95"]
        optional = ["lora_dropout", "augmentation_level", "weight_decay",
                    "alpha_ratio", "dataset_name", "batch_size", "lora_targets",
                    "copy_paste", "mosaic", "warmup", "multi_scale"]
        sel = base + [c for c in optional if c in DB_COLUMNS]
        rows = conn.execute(f"""
            SELECT {', '.join(sel)}
            FROM experiments ORDER BY timestamp
        """).fetchall()
        self._json_response([dict(r) for r in rows])

    def _api_running(self, conn):
        """Return currently running experiments with duration."""
        base_cols = ["id", "run_id", "timestamp", "num_epochs",
                     "learning_rate", "num_train_images"]
        optional = ["lora_rank", "train_image_set", "dataset_name", "lora_dropout",
                    "augmentation_level", "weight_decay", "alpha_ratio",
                    "batch_size", "lora_targets", "copy_paste", "mosaic",
                    "warmup", "multi_scale",
                    "current_epoch", "current_loss", "current_map",
                    "best_epoch", "best_val_map", "loss_history",
                    "model_variant", "method", "grad_accum_steps"]
        sel_cols = base_cols + [c for c in optional if c in DB_COLUMNS]
        rows = conn.execute(f"""
            SELECT {', '.join(sel_cols)}
            FROM experiments WHERE status = 'running'
            ORDER BY timestamp
        """).fetchall()

        running = []
        for r in rows:
            rd = dict(r)
            # Compute how long it's been running (timestamps are naive UTC)
            try:
                started = datetime.fromisoformat(rd["timestamp"])
                rd["running_seconds"] = (datetime.now(timezone.utc).replace(tzinfo=None) - started).total_seconds()
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

    def _api_datasets(self, conn):
        """Return dataset names with per-dataset stats (benchmark mode only)."""
        if not IS_BENCHMARK:
            return self._json_response({"datasets": [], "is_benchmark": False})

        rows = conn.execute("""
            SELECT dataset_name,
                   dataset_num_classes,
                   COUNT(*) as n_completed,
                   AVG(mAP_50) as avg_mAP_50,
                   AVG(mAP_50_95) as avg_mAP_50_95,
                   AVG(train_time_seconds) as avg_train_time,
                   MIN(mAP_50) as min_mAP_50,
                   MAX(mAP_50) as max_mAP_50
            FROM experiments WHERE status = 'completed'
            GROUP BY dataset_name
            ORDER BY dataset_name
        """).fetchall()

        datasets = []
        for r in rows:
            d = dict(r)
            d["avg_mAP_50"] = round(d["avg_mAP_50"] * 100, 1) if d["avg_mAP_50"] else 0
            d["avg_mAP_50_95"] = round(d["avg_mAP_50_95"] * 100, 1) if d["avg_mAP_50_95"] else 0
            datasets.append(d)

        self._json_response({"datasets": datasets, "is_benchmark": True})

    def _api_benchmark_heatmap(self, conn, qs):
        """Config × Dataset heatmap: rows = configs, columns = datasets, cells = mAP_50."""
        if not IS_BENCHMARK:
            return self._json_response({"configs": [], "datasets": [], "cells": []})

        metric = qs.get("metric", ["mAP_50"])[0]
        group_detr_filter = qs.get("group_detr", [None])[0]
        col = {"mAP_50": "mAP_50", "mAP_50_95": "mAP_50_95"}.get(metric, "mAP_50")

        where = ["status = 'completed'"]
        if group_detr_filter and "group_detr" in DB_COLUMNS:
            where.append(f"group_detr = {int(group_detr_filter)}")
        where_sql = " AND ".join(where)

        # Dynamic config columns for heatmap grouping
        hm_cfg_fields = ["lora_rank", "num_epochs", "learning_rate",
                         "lora_dropout", "augmentation_level", "weight_decay", "alpha_ratio"]
        for extra in ["group_detr", "batch_size", "lora_targets", "copy_paste", "mosaic", "warmup", "multi_scale"]:
            if extra in DB_COLUMNS:
                hm_cfg_fields.append(extra)
        cfg_cols_sql = ", ".join(hm_cfg_fields)

        rows = conn.execute(f"""
            SELECT dataset_name, {cfg_cols_sql}, {col} as metric_val
            FROM experiments WHERE {where_sql}
            ORDER BY dataset_name
        """).fetchall()

        # Build config × dataset matrix
        dataset_names = sorted(set(r["dataset_name"] for r in rows))
        config_keys = sorted(set(
            tuple(r[f] for f in hm_cfg_fields) for r in rows
        ))

        def _cfg_key_str(vals):
            parts = []
            for f, v in zip(hm_cfg_fields, vals):
                short = f.replace("lora_", "").replace("augmentation_level", "aug")
                short = short.replace("learning_rate", "lr").replace("weight_decay", "wd")
                short = short.replace("alpha_ratio", "ar").replace("num_epochs", "e")
                short = short.replace("group_detr", "g").replace("batch_size", "bs")
                short = short.replace("copy_paste", "cp").replace("multi_scale", "ms")
                short = short.replace("lora_targets", "lt").replace("warmup", "wu")
                parts.append(f"{short}{v}")
            return "_".join(parts)

        cells = {}
        for r in rows:
            vals = tuple(r[f] for f in hm_cfg_fields)
            cfg_key = _cfg_key_str(vals)
            cells[f"{cfg_key}|{r['dataset_name']}"] = r["metric_val"]

        configs = []
        for ck in config_keys:
            cfg = {"key": _cfg_key_str(ck)}
            for f, v in zip(hm_cfg_fields, ck):
                cfg[f] = v
            configs.append(cfg)

        self._json_response({
            "configs": configs,
            "datasets": dataset_names,
            "cells": cells,
            "metric": metric,
        })

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

    def _api_config_detail(self, conn, qs):
        """Return cross-dataset detail for a specific config (for clickable config cards)."""
        rank = int(qs.get("rank", [0])[0])
        bs = int(qs.get("bs", [0])[0])
        cp = int(qs.get("cp", [0])[0])
        mo = int(qs.get("mo", [0])[0])
        epochs = int(qs.get("epochs", [50])[0])

        experiments = conn.execute("""
            SELECT * FROM experiments
            WHERE lora_rank=? AND batch_size=? AND copy_paste=? AND mosaic=?
              AND num_epochs=? AND status='completed'
            ORDER BY mAP_50_95 DESC
        """, (rank, bs, cp, mo, epochs)).fetchall()

        result = []
        for exp in experiments:
            exp_d = dict(exp)
            evals = conn.execute(
                "SELECT * FROM eval_results WHERE experiment_id=?", (exp_d["id"],)
            ).fetchall()
            exp_d["eval_results"] = [dict(r) for r in evals]
            result.append(exp_d)

        # Load viz_champion images from manifest
        viz_images = {}  # dataset_name -> {pred_image, gt_image, hybrid_image}
        viz_dir = DB_PATH.parent / "viz_champion"
        manifest_path = viz_dir / "manifest.json"
        if manifest_path.exists():
            try:
                import json
                manifest = json.load(open(manifest_path))
                for m in manifest.get("models", []):
                    for ds in m.get("datasets", []):
                        ds_name = ds.get("name") or ds.get("dataset_name")
                        if not ds_name:
                            continue
                        imgs = {}
                        for img_key in ("pred_image", "gt_image", "hybrid_image"):
                            if img_key in ds and ds[img_key]:
                                imgs[img_key] = ds[img_key]
                        if imgs:
                            viz_images[ds_name] = imgs
            except Exception:
                pass

        self._json_response({"experiments": result, "viz_images": viz_images})

    def _api_ft_comparison(self, conn, qs):
        """Compare fine-tune baseline results with LoRA results from --compare-db.

        Returns per-dataset best mAP for each method.
        """
        compare_conn = get_compare_db()

        # Get fine-tune results from the primary DB
        # Detect available columns
        _ft_cols = {c[1] for c in conn.execute("PRAGMA table_info(experiments)").fetchall()}
        _has_method = "method" in _ft_cols
        _has_notes = "notes" in _ft_cols
        _method_col = ", method" if _has_method else ""
        _notes_col = ", notes" if _has_notes else ""
        ft_rows = conn.execute(f"""
            SELECT dataset_name, model_variant, mAP_50, mAP_50_95,
                   train_time_seconds, num_train_images, dataset_num_classes,
                   status, current_epoch, num_epochs{_method_col}{_notes_col}
            FROM experiments
            ORDER BY dataset_name, model_variant
        """).fetchall()

        ft_by_dataset = {}
        lora_by_dataset = {}
        for r in ft_rows:
            r = dict(r)
            ds = r["dataset_name"]
            method = r.get("method", "full_finetune") or "full_finetune"
            variant = r.get("model_variant", "unknown")

            if ds not in ft_by_dataset:
                ft_by_dataset[ds] = {
                    "num_train_images": r["num_train_images"],
                    "dataset_num_classes": r["dataset_num_classes"],
                }

            # LoRA re-eval results live in the same DB — keyed by model_variant
            if method == "lora":
                lora_variant = variant  # e.g. "rfdetr-base" or "rfdetr-nano"
                lora_key = f"lora_{lora_variant.replace('rfdetr-', '')}"  # "lora_base" or "lora_nano"
                entry = {}
                if r["status"] == "completed" and r["mAP_50_95"] is not None:
                    # Parse trial spread from notes JSON if available
                    trial_info = None
                    notes = r.get("notes")
                    if notes:
                        try:
                            trial_info = json.loads(notes)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    entry = {
                        "best_mAP_50": r["mAP_50"],
                        "best_mAP_50_95": r["mAP_50_95"],
                        "best_train_time": r["train_time_seconds"],
                        "best_num_epochs": r["num_epochs"],
                        "num_experiments": 1,
                    }
                    if trial_info:
                        entry["trial_maps"] = trial_info.get("trial_maps")
                        entry["n_trials"] = trial_info.get("n_trials")
                elif r["status"] == "running":
                    entry = {
                        "best_mAP_50": None,
                        "best_mAP_50_95": None,
                        "best_train_time": None,
                        "best_num_epochs": r["num_epochs"],
                        "num_experiments": 1,
                        "status": "running",
                        "current_epoch": r.get("current_epoch"),
                    }
                if entry:
                    if ds not in lora_by_dataset:
                        lora_by_dataset[ds] = {}
                    lora_by_dataset[ds][lora_key] = entry
            else:
                ft_by_dataset[ds][variant] = {
                    "mAP_50": r["mAP_50"],
                    "mAP_50_95": r["mAP_50_95"],
                    "train_time_seconds": r["train_time_seconds"],
                    "status": r["status"],
                    "current_epoch": r["current_epoch"],
                    "num_epochs": r["num_epochs"],
                }

        # Get LoRA results from comparison DB for datasets not yet re-evaluated
        if compare_conn:
            try:
                # Get best LoRA run per dataset (by mAP_50_95)
                lora_cols = [c[1] for c in compare_conn.execute(
                    "PRAGMA table_info(experiments)").fetchall()]
                has_lora_rank = "lora_rank" in lora_cols
                lora_rows = compare_conn.execute("""
                    WITH best AS (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY dataset_name ORDER BY mAP_50_95 DESC
                        ) as rn
                        FROM experiments WHERE status='completed'
                    ),
                    counts AS (
                        SELECT dataset_name, COUNT(*) as num_experiments
                        FROM experiments WHERE status='completed'
                        GROUP BY dataset_name
                    )
                    SELECT b.dataset_name, b.mAP_50 as best_mAP_50,
                           b.mAP_50_95 as best_mAP_50_95,
                           b.train_time_seconds as best_train_time,
                           b.num_epochs as best_num_epochs,
                           c.num_experiments
                    FROM best b JOIN counts c ON b.dataset_name = c.dataset_name
                    WHERE b.rn = 1
                """).fetchall()
                for r in lora_rows:
                    r = dict(r)
                    ds = r["dataset_name"]
                    # Only show corrected LoRA results — skip old uncorrected data
                    pass
            except Exception as e:
                logger.warning("Failed to read compare DB: %s", e)

        # Merge into comparison table
        all_datasets = sorted(set(list(ft_by_dataset.keys()) + list(lora_by_dataset.keys())))
        comparison = []
        for ds in all_datasets:
            lora_data = lora_by_dataset.get(ds, {})
            row = {
                "dataset_name": ds,
                "num_train_images": ft_by_dataset.get(ds, {}).get("num_train_images"),
                "dataset_num_classes": ft_by_dataset.get(ds, {}).get("dataset_num_classes"),
                "ft_nano": ft_by_dataset.get(ds, {}).get("rfdetr-nano"),
                "ft_medium": ft_by_dataset.get(ds, {}).get("rfdetr-medium"),
                "ft_2xlarge": ft_by_dataset.get(ds, {}).get("rfdetr-2xlarge"),
                "lora_medium": lora_data.get("lora_medium"),
                "lora_nano": lora_data.get("lora_nano"),
                "lora_2xlarge": lora_data.get("lora_2xlarge"),
            }
            comparison.append(row)

        self._json_response({
            "comparison": comparison,
            "has_compare_db": compare_conn is not None,
        })

    def _api_viz_champion(self):
        """Build multi-model visualization manifest from DB + any existing images.

        Each model that won at least 1 dataset gets ALL its dataset results (not just wins).
        """
        try:
            conn = get_db()
            from collections import OrderedDict

            # Step 1: Find which config keys won at least 1 dataset
            winner_rows = conn.execute("""
                WITH ranked AS (
                    SELECT *, RANK() OVER (PARTITION BY dataset_name ORDER BY mAP_50_95 DESC) as rnk
                    FROM experiments WHERE status='completed'
                )
                SELECT DISTINCT lora_rank, batch_size, copy_paste, mosaic, alpha_ratio, weight_decay,
                       lora_alpha, num_epochs, learning_rate, lora_targets
                FROM ranked WHERE rnk = 1
            """).fetchall()

            if not winner_rows:
                self._json_response({"error": "No completed experiments yet.", "ready": False})
                return

            # Build set of winning config keys
            winning_configs = {}
            for r in winner_rows:
                key = f"{r['lora_rank']}_{r['batch_size']}_{r['copy_paste']}_{r['mosaic']}_{r['alpha_ratio']}_{r['weight_decay']}_{r['num_epochs']}"
                winning_configs[key] = {
                    "rank": r["lora_rank"], "batch_size": r["batch_size"],
                    "copy_paste": bool(r["copy_paste"]), "mosaic": bool(r["mosaic"]),
                    "alpha_ratio": r["alpha_ratio"], "weight_decay": r["weight_decay"],
                    "alpha": r["lora_alpha"], "num_epochs": r["num_epochs"], "epochs": r["num_epochs"],
                    "lr": r["learning_rate"], "lora_targets": r["lora_targets"],
                }

            # Step 2: Get ALL results for winning configs, with ranking per dataset
            all_rows = conn.execute("""
                WITH ranked AS (
                    SELECT *, RANK() OVER (PARTITION BY dataset_name ORDER BY mAP_50_95 DESC) as rnk,
                           COUNT(*) OVER (PARTITION BY dataset_name) as total_per_ds
                    FROM experiments WHERE status='completed'
                )
                SELECT dataset_name, lora_rank, batch_size, copy_paste, mosaic, alpha_ratio,
                       weight_decay, lora_alpha, num_epochs, learning_rate, lora_targets,
                       mAP_50, mAP_50_95, train_time_seconds,
                       dataset_num_classes, num_train_images, total_per_ds, rnk
                FROM ranked
                ORDER BY dataset_name, rnk
            """).fetchall()

            # Group by config key, only keep winning configs
            config_groups = OrderedDict()
            for r in all_rows:
                key = f"{r['lora_rank']}_{r['batch_size']}_{r['copy_paste']}_{r['mosaic']}_{r['alpha_ratio']}_{r['weight_decay']}_{r['num_epochs']}"
                if key not in winning_configs:
                    continue
                if key not in config_groups:
                    config_groups[key] = {"config": winning_configs[key], "datasets": [], "wins": 0}
                config_groups[key]["datasets"].append(dict(r))
                if r["rnk"] == 1:
                    config_groups[key]["wins"] += 1

            # Load existing manifest for image paths
            viz_dir = DB_PATH.parent / "viz_champion"
            manifest_path = viz_dir / "manifest.json"
            existing_images = {}  # (model_index, dataset_name) -> {pred_image, ...}
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        old = json.load(f)
                    if "models" in old:
                        for m in old["models"]:
                            mi = m.get("model_index", 0)
                            for ds in m.get("datasets", []):
                                existing_images[(mi, ds["name"])] = ds
                    elif "datasets" in old:
                        for ds in old["datasets"]:
                            existing_images[(0, ds["name"])] = ds
                except Exception:
                    pass

            # Build models list sorted by wins desc
            models = []
            for key, grp in sorted(config_groups.items(), key=lambda kv: -kv[1]["wins"]):
                dsets = []
                for r in grp["datasets"]:
                    ds_name = r["dataset_name"]
                    is_winner = r["rnk"] == 1
                    entry = {
                        "name": ds_name,
                        "num_classes": r["dataset_num_classes"],
                        "num_train_images": r["num_train_images"],
                        "mAP_50": r["mAP_50"],
                        "mAP_50_95": r["mAP_50_95"],
                        "train_time_seconds": r["train_time_seconds"],
                        "ranking": r["rnk"],
                        "total_configs": r["total_per_ds"],
                        "is_winner": is_winner,
                    }
                    # Merge image paths from existing manifest
                    mi = len(models)
                    for lookup_key in [(mi, ds_name), (0, ds_name)]:
                        if lookup_key in existing_images:
                            ei = existing_images[lookup_key]
                            for img_key in ("pred_image", "gt_image", "hybrid_image",
                                            "original_image", "test_image_stem",
                                            "num_gt_boxes", "num_predictions",
                                            "num_test_images", "class_names",
                                            "raw_detections", "gt_boxes_data",
                                            "base_coco_detections", "image_size",
                                            "sample_mAP_50", "sample_mAP_50_95"):
                                if img_key in ei:
                                    entry[img_key] = ei[img_key]
                            break
                    dsets.append(entry)

                n_wins = grp["wins"]
                avg_map50 = sum(d["mAP_50"] or 0 for d in dsets) / len(dsets)
                avg_map5095 = sum(d["mAP_50_95"] or 0 for d in dsets) / len(dsets)
                avg_time = sum(d["train_time_seconds"] or 0 for d in dsets) / len(dsets)

                models.append({
                    "model_index": len(models),
                    "config_key": key,
                    "config": grp["config"],
                    "wins": n_wins,
                    "datasets": dsets,
                    "avg_mAP_50": avg_map50,
                    "avg_mAP_50_95": avg_map5095,
                    "avg_train_time": avg_time,
                })

            self._json_response({
                "metric": "mAP_50_95",
                "models": models,
                "ready": True,
            })
        except Exception as e:
            logging.exception("Error building viz manifest")
            self._json_response({"error": str(e), "ready": False})

    def _serve_viz_file(self, path):
        """Serve files from viz_champion/ directory under /viz/ URL prefix."""
        import mimetypes
        # /viz/images/foo.jpg or /viz_champion/images/foo.jpg → viz_champion/images/foo.jpg
        if path.startswith("/viz_champion/"):
            rel = path[len("/viz_champion/"):]
        else:
            rel = path[len("/viz/"):]
        file_path = DB_PATH.parent / "viz_champion" / rel
        if not file_path.exists() or ".." in rel:
            self.send_error(404)
            return
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results.db",
                        help="Database file (e.g. results.db or results_phase2.db)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="",
                        help="Bind address (default: '' = localhost, use '0.0.0.0' for all interfaces)")
    parser.add_argument("--remote", type=str, default=None,
                        help="Remote host for DB sync, e.g. roboflow@100.94.130.94")
    parser.add_argument("--remote-db", type=str, default=None,
                        help="Path to DB on remote host (default: auto-detect from --db)")
    parser.add_argument("--compare-db", type=str, default=None,
                        help="Second DB for LoRA vs FT comparison (e.g. results_phase3b.db)")
    parser.add_argument("--sync-interval", type=int, default=10,
                        help="Seconds between background syncs (default: 10)")
    args = parser.parse_args()

    DB_PATH = Path(__file__).parent / args.db

    # Set up comparison DB if provided
    if args.compare_db:
        COMPARE_DB_PATH = Path(__file__).parent / args.compare_db
        if COMPARE_DB_PATH.exists():
            print(f"📊 Comparison DB: {COMPARE_DB_PATH}")
        else:
            print(f"⚠️  Comparison DB not found: {COMPARE_DB_PATH}")

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

    # Auto-detect phase 2 / benchmark and available columns
    global DB_COLUMNS
    DB_COLUMNS = set()
    if DB_PATH.exists():
        _conn = sqlite3.connect(str(DB_PATH))
        _conn.row_factory = sqlite3.Row
        DB_COLUMNS = {c[1] for c in _conn.execute("PRAGMA table_info(experiments)").fetchall()}
        IS_BENCHMARK = _detect_benchmark(_conn)
        IS_PHASE2 = _detect_phase2(_conn)
        _conn.close()
        if IS_BENCHMARK:
            print(f"📊 Benchmark DB detected: {DB_PATH}")
        elif IS_PHASE2:
            print(f"📊 Phase 2 DB detected: {DB_PATH}")

    port = args.port
    host = args.host
    sync_info = f" · syncing from {args.remote} every {args.sync_interval}s" if args.remote else ""
    display_host = host or "localhost"
    print(f"🚀 Results explorer: http://{display_host}:{port}{sync_info}")
    # Use HTTP/1.1 to avoid browser connection queueing issues
    Handler.protocol_version = "HTTP/1.1"

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

        def handle_error(self, request, client_address):
            """Silently ignore client disconnection errors instead of crashing."""
            import traceback
            exc_type = sys.exc_info()[0]
            if exc_type in (ConnectionResetError, BrokenPipeError,
                            ConnectionAbortedError, OSError):
                return  # harmless — client went away
            # Log other errors but don't crash
            print(f"[server] Error handling request from {client_address}:")
            traceback.print_exc()

    ThreadedHTTPServer((host, port), Handler).serve_forever()
