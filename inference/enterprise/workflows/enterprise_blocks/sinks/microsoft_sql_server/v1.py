import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, field_validator

if TYPE_CHECKING:
    import pyodbc
else:
    try:
        import pyodbc
    except ImportError:
        pyodbc = None

PYODBC_AVAILABLE = pyodbc is not None

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

logger = logging.getLogger(__name__)


class SQLServerError(Exception):
    """Base exception for SQL Server related errors"""

    pass


class SQLServerConnectionError(SQLServerError):
    """Exception raised for connection-related errors"""

    pass


class SQLServerInsertError(SQLServerError):
    """Exception raised for insert operation errors"""

    pass


LONG_DESCRIPTION = """
The **Microsoft SQL Server Sink** block enables users to send data from a Roboflow workflow directly to a Microsoft SQL Server 
database. This block allows seamless integration of inference results, metadata, and processed data into structured SQL 
databases for further analysis, reporting, or automation.

### Database Connection Setup

The block supports two authentication methods:

1. **Windows Authentication (Default)**: Uses the current Windows credentials
2. **SQL Server Authentication**: Uses username and password

Required connection parameters:
* **Host**: The IP address or hostname of the Microsoft SQL Server instance
* **Port**: The port number for SQL Server (default: 1433)
* **Database**: The target database where data will be inserted
* **Table Name**: The name of the table where the data will be inserted

Optional authentication parameters (for SQL Server Authentication):
* **Username**: The SQL Server username for authentication
* **Password**: The password associated with the username

If username and password are not provided, the block will use Windows Authentication (trusted connection).

### Data Input Format

The block expects data in a dictionary format or list of dictionaries that map to the target table columns:

```python
# Single row
{
    "timestamp": "2025-02-12T10:30:00Z",
    "part_detected": "Defective Part",
    "confidence": 0.92,
    "camera_id": "CAM_001"
}

# Multiple rows
[
    {
        "timestamp": "2025-02-12T10:30:00Z",
        "part_detected": "Defective Part",
        "confidence": 0.92,
        "camera_id": "CAM_001"
    },
    {
        "timestamp": "2025-02-12T10:31:00Z",
        "part_detected": "Good Part",
        "confidence": 0.95,
        "camera_id": "CAM_002"
    }
]
```

### Important Notes

* The specified table must already exist in the database
* The authenticated user must have INSERT permissions
* Column names in the data must match the table schema
* When using Windows Authentication, ensure the service account has proper permissions
* The pyodbc package must be installed
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Microsoft SQL Server Sink",
            "version": "v1",
            "short_description": "Save data to a Microsoft SQL Server database.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-database",
                "blockPriority": 3,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/microsoft_sql_server_sink@v1"]

    host: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="SQL Server host address",
        examples=["localhost", "192.168.1.100"],
    )
    port: Union[Selector(kind=[STRING_KIND]), int] = Field(
        default=1433,
        description="SQL Server port",
        examples=[1433],
    )
    database: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Target database name",
        examples=["production_db"],
    )
    username: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="SQL Server username",
        examples=["db_user"],
    )
    password: Optional[Union[Selector(kind=[SECRET_KIND]), str]] = Field(
        default=None,
        description="SQL Server password",
        examples=["$inputs.sql_password"],
    )
    table_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Target table name",
        examples=["detections"],
    )

    data: Union[
        Selector(kind=[DICTIONARY_KIND]), Union[Dict[str, Any], List[Dict[str, Any]]]
    ] = Field(
        description="Data to insert into the database. Can be a single dictionary or list of dictionaries.",
        examples=[
            {"timestamp": "2025-02-12T10:30:00Z", "object_detected": "Defective Part"},
            [
                {
                    "timestamp": "2025-02-12T10:30:00Z",
                    "object_detected": "Defective Part",
                },
                {"timestamp": "2025-02-12T10:31:00Z", "object_detected": "Good Part"},
            ],
        ],
    )

    fire_and_forget: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True,
        description="Run in asynchronous mode for faster processing",
        examples=[True, "$inputs.fire_and_forget"],
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, value: Any) -> Any:
        if isinstance(value, int) and not (1 <= value <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MicrosoftSQLServerSinkBlockV1(WorkflowBlock):
    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._connection = None
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        host: str,
        port: int,
        database: str,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        username: Optional[str] = None,
        password: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> BlockResult:
        registration_task = partial(
            self._process_data,
            host=host,
            port=port,
            database=database,
            table_name=table_name,
            data=data,
            username=username,
            password=password,
        )
        if fire_and_forget and self._background_tasks is not None:
            self._background_tasks.add_task(registration_task)
            return {
                "error_status": False,
                "message": "Data processing scheduled",
            }
        elif fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(registration_task)
            return {
                "error_status": False,
                "message": "Data processing scheduled",
            }
        else:
            return registration_task()

    def _process_data(
        self,
        host: str,
        port: int,
        database: str,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            with self._get_connection(
                host, port, database, username, password
            ) as connection:
                data_list = self._validate_data(data)
                self._insert_data(connection, table_name, data_list)
                return {
                    "error_status": False,
                    "message": f"Successfully inserted {len(data_list)} records",
                }
        except SQLServerError as e:
            return {
                "error_status": True,
                "message": str(e),
            }
        except Exception as e:
            logger.error(f"Unexpected error in SQL Server sink: {str(e)}")
            return {
                "error_status": True,
                "message": f"An unexpected error occurred: {str(e)}",
            }

    @contextmanager
    def _get_connection(
        self,
        host: str,
        port: int,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        connection = None
        try:
            connection = self._create_connection(
                host, port, database, username, password
            )
            yield connection
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {str(e)}")

    def _create_connection(
        self,
        host: str,
        port: int,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        if not PYODBC_AVAILABLE:
            raise SQLServerConnectionError(
                "pyodbc package is not installed. Please contact Roboflow's Enterprise support team for assistance."
            )

        connection_string = (
            f"DRIVER={{FreeTDS}};"
            f"SERVER={host};"
            f"PORT={port};"
            f"DATABASE={database};"
        )

        if username and password:
            connection_string += f"UID={username};PWD={password}"
        else:
            connection_string += "Trusted_Connection=yes"

        try:
            connection = pyodbc.connect(connection_string, autocommit=False)
            cursor = connection.cursor()
            try:
                cursor.execute("SET ANSI_NULLS ON")
                cursor.execute("SET ANSI_PADDING ON")
                cursor.execute("SET ANSI_WARNINGS ON")
                cursor.execute("SET ARITHABORT ON")
                cursor.execute("SET QUOTED_IDENTIFIER ON")
                connection.commit()
            except pyodbc.Error as e:
                connection.rollback()
                cursor.close()
                raise SQLServerError(
                    f"Failed to set required session parameters: {str(e)}"
                )
            finally:
                cursor.close()
            return connection
        except pyodbc.Error as e:
            raise SQLServerConnectionError(str(e))

    def _insert_data(
        self, connection: Any, table_name: str, data: List[Dict[str, Any]]
    ) -> None:
        if not data:
            return

        try:
            columns = list(data[0].keys())
            placeholders = ",".join(["?" for _ in columns])
            column_names = ",".join(columns)

            query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

            cursor = connection.cursor()
            try:
                for row in data:
                    values = [row[col] for col in columns]
                    cursor.execute(query, values)
                connection.commit()
            except pyodbc.DataError as e:
                connection.rollback()
                raise SQLServerInsertError(f"Data conversion error: {str(e)}")
            except pyodbc.Error as e:
                connection.rollback()
                raise SQLServerInsertError(f"Failed to insert data: {str(e)}")
            finally:
                cursor.close()
        except Exception as e:
            raise SQLServerInsertError(f"Error preparing or executing insert: {str(e)}")

    def _validate_data(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            return [data]

        if isinstance(data, list):
            if not data:
                raise ValueError("Empty list provided for insert operation")

            if not all(isinstance(item, dict) for item in data):
                raise ValueError("All items in data list must be dictionaries")

            if len(data) > 1:
                first_keys = set(data[0].keys())
                for idx, item in enumerate(data[1:], 1):
                    if set(item.keys()) != first_keys:
                        raise ValueError(
                            f"Dictionary at index {idx} has different keys than the first dictionary"
                        )

            return data

    def __del__(self):
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                logger.error(f"Error closing connection in destructor: {str(e)}")
