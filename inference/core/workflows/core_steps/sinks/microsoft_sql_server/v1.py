import json
import logging
from typing import Any, Dict, List, Literal, Optional, Type, Union

try:
    import pyodbc
except ImportError:
    import subprocess
    import sys
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyodbc"])
    import pyodbc
from pydantic import ConfigDict, Field, field_validator

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **Microsoft SQL Server Sink** block enables users to send data from a Roboflow Workflow directly to a Microsoft SQL Server 
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

The block expects data in a structured JSON format that maps to the target table columns:

```json
[
    {
        "timestamp": "2025-02-12T10:30:00Z",
        "object_detected": "Defective Part",
        "confidence": 0.92,
        "camera_id": "CAM_001"
    }
]
```

### Important Notes

* The specified table must already exist in the database
* The authenticated user must have INSERT permissions
* Column names in the data must match the table schema
* The block operates in fire-and-forget mode
* When using Windows Authentication, ensure the service account has proper permissions

!!! warning "Security considerations"

    * Store credentials securely and never hardcode sensitive information
    * Ensure proper network security and firewall rules
    * Use the minimum required permissions for the database user
    * Consider using environment variables for sensitive connection details
    * Windows Authentication is generally more secure than SQL Authentication
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
    
    # Connection parameters
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
        description="SQL Server username (optional - if not provided, Windows Authentication will be used)",
        examples=["db_user"],
    )
    password: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="SQL Server password (optional - if not provided, Windows Authentication will be used)",
        examples=["password123"],
    )
    table_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Target table name",
        examples=["detections"],
    )
    
    # Data input
    data: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="JSON data to insert into the database",
        examples=['[{"timestamp": "2025-02-12T10:30:00Z", "object_detected": "Defective Part"}]'],
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
    def __init__(self):
        self._connection = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _create_connection(
        self,
        host: str,
        port: int,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        # Base connection string
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
        )
        
        # Add authentication details
        if username and password:
            # SQL Server Authentication
            connection_string += f"UID={username};PWD={password}"
        else:
            # Windows Authentication
            connection_string += "Trusted_Connection=yes"
        
        try:
            self._connection = pyodbc.connect(connection_string)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SQL Server: {str(e)}")

    def _insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        if not data:
            return

        # Extract column names from the first data item
        columns = list(data[0].keys())
        placeholders = ",".join(["?" for _ in columns])
        column_names = ",".join(columns)
        
        # Prepare the SQL query
        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
        
        cursor = self._connection.cursor()
        try:
            # Insert each row of data
            for row in data:
                values = [row[col] for col in columns]
                cursor.execute(query, values)
            
            self._connection.commit()
        finally:
            cursor.close()

    def run(
        self,
        host: str,
        port: int,
        database: str,
        table_name: str,
        data: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> BlockResult:
        try:
            # Parse the JSON data
            parsed_data = json.loads(data)
            if not isinstance(parsed_data, list):
                parsed_data = [parsed_data]

            # Create database connection
            self._create_connection(
                host=host,
                port=port,
                database=database,
                username=username,
                password=password,
            )

            # Insert the data
            self._insert_data(table_name=table_name, data=parsed_data)

            return {
                "error_status": False,
                "message": f"Successfully inserted {len(parsed_data)} rows into {table_name}",
            }

        except json.JSONDecodeError as e:
            return {
                "error_status": True,
                "message": f"Invalid JSON data format: {str(e)}",
            }
        except Exception as e:
            return {"error_status": True, "message": str(e)}
        finally:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __del__(self):
        if self._connection is not None:
            self._connection.close()
