import datetime
import os
import sqlite3
from typing import Dict, Optional, Union

import requests

from inference.core.env import MODEL_CACHE_DIR
from inference.core.logger import logger
from inference.core.utils.sqlite_wrapper import SQLiteWrapper
from inference.usage_tracking.payload_helpers import APIKey, APIKeyHash, sha256_hash


class UsageProfile(SQLiteWrapper):
    def __init__(
        self,
        api_plan_endpoint_url: str,
        db_file_path: str = os.path.join(MODEL_CACHE_DIR, "usage.db"),
        table_name: str = "plan_details",
        sqlite_connection: Optional[sqlite3.Connection] = None,
        cache_enabled: bool = True,
    ):
        self._api_plan_endpoint_url = api_plan_endpoint_url

        self._columns = {}

        # "YYYY-MM-DD HH:MM:SS" UTC / https://www.sqlite.org/lang_createtable.html#the_default_clause
        self._ts_format = "%Y-%m-%d %H:%M:%S"
        self._ts_col_name = "ts"
        self._columns[self._ts_col_name] = "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"

        self._api_key_hash_col_name = "api_key_hash"
        self._columns[self._api_key_hash_col_name] = "TEXT NOT NULL"

        self._is_enterprise_default = True
        self._is_enterprise_col_name = "is_enterprise"
        self._columns[self._is_enterprise_col_name] = (
            f"INTEGER NOT NULL DEFAULT {1 if self._is_enterprise_default else 0}"
        )

        self._is_pro_default = True
        self._is_pro_col_name = "is_pro"
        self._columns[self._is_pro_col_name] = (
            f"INTEGER NOT NULL DEFAULT {1 if self._is_pro_default else 0}"
        )

        self._is_trial_default = False
        self._is_trial_col_name = "is_trial"
        self._columns[self._is_trial_col_name] = (
            f"INTEGER NOT NULL DEFAULT {1 if self._is_trial_default else 0}"
        )

        self._is_billed_default = True
        self._is_billed_col_name = "is_billed"
        self._columns[self._is_billed_col_name] = (
            f"INTEGER NOT NULL DEFAULT {1 if self._is_billed_default else 0}"
        )

        self._offline_enabled_default = True
        self._offline_enabled_col_name = "offline_enabled"
        self._columns[self._offline_enabled_col_name] = (
            f"INTEGER NOT NULL DEFAULT {1 if self._offline_enabled_default else 0}"
        )

        self._over_quota_default = False
        self._over_quota_col_name = "over_quota"
        self._columns[self._over_quota_col_name] = (
            f"INTEGER NOT NULL DEFAULT {1 if self._over_quota_default else 0}"
        )

        self._usage_profiles: Dict[APIKeyHash, Dict[str, Union[str, bool]]] = {}
        self._cache_enabled = cache_enabled
        if cache_enabled:
            super().__init__(
                db_file_path=db_file_path,
                table_name=table_name,
                columns=self._columns,
                connection=sqlite_connection,
            )
            self._usage_profiles = self.from_cache(sqlite_connection=sqlite_connection)

    def from_cache(self, sqlite_connection: Optional[sqlite3.Connection]):
        if not self._cache_enabled:
            return

        try:
            cached_usage_profiles = self.select(
                connection=sqlite_connection, with_exclusive=True
            )
        except Exception as exc:
            logger.debug("Could not obtain cached usage profiles - %s", exc)
            return

        for cached_usage_profile in cached_usage_profiles:
            api_key_hash = cached_usage_profile[self._api_key_hash_col_name]
            if api_key_hash in self._usage_profiles:
                continue
            self._usage_profiles[api_key_hash] = cached_usage_profile

    def refresh_usage_profile(self, api_key: APIKey):
        ssl_verify = True
        if "localhost" in self._api_plan_endpoint_url.lower():
            ssl_verify = False
        if "127.0.0.1" in self._api_plan_endpoint_url.lower():
            ssl_verify = False

        api_key_hash = sha256_hash(api_key)
        if api_key_hash in self._usage_profiles:
            usage_profile = self._usage_profiles[api_key_hash]
        else:
            usage_profile = {
                self._ts_col_name: "1970-01-01 00:00:00",
                self._api_key_hash_col_name: api_key_hash,
                self._is_enterprise_col_name: self._is_enterprise_default,
                self._is_pro_col_name: self._is_pro_default,
                self._is_trial_col_name: self._is_trial_default,
                self._is_billed_col_name: self._is_billed_default,
                self._over_quota_col_name: self._over_quota_default,
            }

        try:
            response = requests.get(
                self._api_plan_endpoint_url,
                verify=ssl_verify,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=1,
            )
            if response.status_code != 200:
                logger.debug(
                    "Got %s from %s", response.status_code, self._api_plan_endpoint_url
                )
                if (
                    400 <= response.status_code < 500
                    and api_key_hash not in self._usage_profiles
                ):
                    usage_profile[self._is_enterprise_col_name] = False
                    usage_profile[self._is_pro_col_name] = False
                    usage_profile[self._is_billed_col_name] = False
                    usage_profile[self._over_quota_col_name] = True
            else:
                usage_profile_from_api: Dict[str, bool] = response.json()
                usage_profile[self._is_enterprise_col_name] = (
                    usage_profile_from_api.get(
                        self._is_enterprise_col_name, self._is_enterprise_default
                    )
                )
                usage_profile[self._is_pro_col_name] = usage_profile_from_api.get(
                    self._is_pro_col_name, self._is_pro_default
                )
                usage_profile[self._is_billed_col_name] = usage_profile_from_api.get(
                    self._is_billed_col_name, self._is_billed_default
                )
                usage_profile[self._over_quota_col_name] = usage_profile_from_api.get(
                    self._over_quota_col_name, self._over_quota_default
                )
                usage_profile[self._is_trial_col_name] = usage_profile_from_api.get(
                    self._is_trial_col_name, self._is_trial_default
                )
                usage_profile[self._ts_col_name] = datetime.datetime.now(
                    tz=datetime.timezone.utc
                ).strftime(format=self._ts_format)
        except requests.exceptions.JSONDecodeError:
            logger.debug("Could not parse usage profile '%s'", response.content)
        except Exception as exc:
            logger.debug(
                "Could not obtain usage profile from %s for %s - %s",
                self._api_plan_endpoint_url,
                api_key,
                exc,
            )

        self._usage_profiles[usage_profile[self._api_key_hash_col_name]] = usage_profile

        if self._cache_enabled:
            pass
