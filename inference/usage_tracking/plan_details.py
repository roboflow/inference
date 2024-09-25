import datetime
import os
import sqlite3
from typing import Dict, Optional, Union

import requests

from inference.core.env import MODEL_CACHE_DIR
from inference.core.logger import logger
from inference.core.utils.sqlite_wrapper import SQLiteWrapper
from inference.usage_tracking.payload_helpers import APIKey, APIKeyHash, sha256_hash


class PlanDetails(SQLiteWrapper):
    def __init__(
        self,
        api_plan_endpoint_url: str,
        db_file_path: str = os.path.join(MODEL_CACHE_DIR, "usage.db"),
        table_name: str = "api_keys_plans",
        sqlite_connection: Optional[sqlite3.Connection] = None,
        sqlite_cache_enabled: bool = True,
    ):
        self._api_plan_endpoint_url = api_plan_endpoint_url
        self._cache_ttl_seconds = 86400

        self._columns = {}

        # "YYYY-MM-DD HH:MM:SS" UTC / https://www.sqlite.org/lang_createtable.html#the_default_clause
        self._ts_format = "%Y-%m-%d %H:%M:%S %Z"
        self._ts_col_name = "ts"
        self._ts_default = "1970-01-01 00:00:00 UTC"
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

        self.api_keys_plans: Dict[APIKeyHash, Dict[str, Union[str, bool]]] = {}
        self._sqlite_cache_enabled = sqlite_cache_enabled
        if sqlite_cache_enabled:
            super().__init__(
                db_file_path=db_file_path,
                table_name=table_name,
                columns=self._columns,
                connection=sqlite_connection,
            )
            self.api_keys_plans = self.load_from_cache(
                sqlite_connection=sqlite_connection
            )

    def load_from_cache(
        self, sqlite_connection: Optional[sqlite3.Connection] = None
    ) -> Dict[APIKeyHash, Dict[str, Union[str, bool]]]:
        if not self._sqlite_cache_enabled:
            return []

        try:
            cached_api_keys_plans = self.select(
                connection=sqlite_connection, with_exclusive=True
            )
        except Exception as exc:
            logger.debug("Could not obtain cached api key plans - %s", exc)
            return

        api_keys_plans = {}
        for cached_api_key_plan in cached_api_keys_plans:
            api_key_hash = cached_api_key_plan[self._api_key_hash_col_name]
            if api_key_hash in api_keys_plans:
                continue
            api_keys_plans[api_key_hash] = cached_api_key_plan
        return api_keys_plans

    def refresh_api_key_plan_cache(
        self, api_key: APIKey, sqlite_connection: Optional[sqlite3.Connection] = None
    ) -> Dict[str, Union[str, bool]]:
        ssl_verify = True
        if "localhost" in self._api_plan_endpoint_url.lower():
            ssl_verify = False
        if "127.0.0.1" in self._api_plan_endpoint_url.lower():
            ssl_verify = False

        api_key_hash = sha256_hash(api_key)
        if api_key_hash in self.api_keys_plans:
            api_key_plan = self.api_keys_plans[api_key_hash]
        else:
            api_key_plan = {
                self._ts_col_name: self._ts_default,
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
                    and api_key_hash not in self.api_keys_plans
                ):
                    api_key_plan[self._is_enterprise_col_name] = False
                    api_key_plan[self._is_pro_col_name] = False
                    api_key_plan[self._is_billed_col_name] = False
                    api_key_plan[self._over_quota_col_name] = True
            else:
                api_key_plan_from_api: Dict[str, bool] = response.json()
                api_key_plan[self._is_enterprise_col_name] = api_key_plan_from_api.get(
                    self._is_enterprise_col_name, self._is_enterprise_default
                )
                api_key_plan[self._is_pro_col_name] = api_key_plan_from_api.get(
                    self._is_pro_col_name, self._is_pro_default
                )
                api_key_plan[self._is_billed_col_name] = api_key_plan_from_api.get(
                    self._is_billed_col_name, self._is_billed_default
                )
                api_key_plan[self._over_quota_col_name] = api_key_plan_from_api.get(
                    self._over_quota_col_name, self._over_quota_default
                )
                api_key_plan[self._is_trial_col_name] = api_key_plan_from_api.get(
                    self._is_trial_col_name, self._is_trial_default
                )
                api_key_plan[self._ts_col_name] = datetime.datetime.now(
                    tz=datetime.timezone.utc
                ).strftime(format=self._ts_format)
        except requests.exceptions.JSONDecodeError:
            logger.debug("Could not parse api key plan '%s'", response.content)
        except Exception as exc:
            logger.debug(
                "Could not obtain api key plan from %s for %s - %s",
                self._api_plan_endpoint_url,
                api_key,
                exc,
            )

        api_key_hash = api_key_plan[self._api_key_hash_col_name]
        self.api_keys_plans[api_key_hash] = api_key_plan

        if self._sqlite_cache_enabled:
            try:
                refreshed_api_key_plan = self.refresh(
                    rows=[api_key_plan], connection=sqlite_connection
                )
                if refreshed_api_key_plan:
                    self.api_keys_plans[api_key_hash] = refreshed_api_key_plan[0]
            except Exception as exc:
                logger.debug("Failed to refresh api key plan cache - %s", exc)

        return self.api_keys_plans[api_key_hash]

    def get_api_key_plan(
        self,
        api_key: APIKey,
        sqlite_connection: Optional[sqlite3.Connection] = None,
        date_time_now: Optional[datetime.datetime] = None,
    ) -> Dict[str, Union[str, bool]]:
        if date_time_now is None:
            date_time_now = datetime.datetime.now(tz=datetime.timezone.utc)
        api_key_hash = sha256_hash(api_key)
        if api_key_hash not in self.api_keys_plans:
            api_key_plan = self.refresh_api_key_plan_cache(
                api_key=api_key, sqlite_connection=sqlite_connection
            )
        else:
            api_key_plan = self.api_keys_plans[api_key_hash]

        cache_ts_str = self._ts_default
        if self._ts_col_name in api_key_plan:
            cache_ts_str = api_key_plan[self._ts_col_name]
        try:
            cache_ts = datetime.datetime.strptime(
                cache_ts_str, self._ts_format
            ).replace(tzinfo=datetime.timezone.utc)
        except Exception as exc:
            logger.debug("Failed to parse api key plan cache timestamp - %s", exc)
            cache_ts = datetime.datetime.strptime(
                self._ts_default, self._ts_format
            ).replace(tzinfo=datetime.timezone.utc)

        cache_age = date_time_now - cache_ts
        if cache_age.seconds + cache_age.days * 86400 > self._cache_ttl_seconds:
            api_key_plan = self.refresh_api_key_plan_cache(
                api_key=api_key, sqlite_connection=sqlite_connection
            )

        return api_key_plan
