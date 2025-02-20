import json
import logging
from typing import Any, Dict, List, Literal, Optional, Type, Union
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

logger = logging.getLogger(__name__)

class SQLServerError(Exception):
    """Base exception for SQL Server related errors"""
    pass

class SQLServerConnectionError(SQLServerError):
    """Exception raised for connection-related errors"""
    pass

class SQLServerAuthenticationError(SQLServerError):
    """Exception raised for authentication-related errors"""
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

The block expects data in a structured JSON format that maps to the target table columns:

```json
[
    {
        "timestamp": "2025-02-12T10:30:00Z",
        "object_detected": "Defective Part",
        "confidence": 0.92,
        "camera_id": "CAM_001"
    },
    ...
]
```

### Important Notes

* The specified table must already exist in the database
* The authenticated user must have INSERT permissions
* Column names in the data must match the table schema
* When using Windows Authentication, ensure the service account has proper permissions
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
    password: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="SQL Server password",
        examples=["password123"],
    )
    table_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Target table name",
        examples=["detections"],
    )
    
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
            self._connection = pyodbc.connect(connection_string, autocommit=False)
            cursor = self._connection.cursor()
            try:
                cursor.execute("SET ANSI_NULLS ON")
                cursor.execute("SET ANSI_PADDING ON")
                cursor.execute("SET ANSI_WARNINGS ON")
                cursor.execute("SET ARITHABORT ON")
                cursor.execute("SET QUOTED_IDENTIFIER ON")
            except pyodbc.Error as e:
                logger.warning(f"Failed to set session options: {str(e)}")
            finally:
                cursor.close()
        except pyodbc.Error as e:
            error_msg = str(e)
            if "Login failed" in error_msg or "password" in error_msg.lower():
                raise SQLServerAuthenticationError(f"Authentication failed: {error_msg}")
            elif "Cannot open database" in error_msg:
                raise SQLServerConnectionError(f"Database access error: {error_msg}")
            elif "Network" in error_msg or "Communication" in error_msg:
                raise SQLServerConnectionError(f"Network/connectivity error: {error_msg}")
            else:
                raise SQLServerConnectionError(f"Failed to connect to SQL Server: {error_msg}")

    def _validate_data(self, data: List[Dict[str, Any]]) -> None:
        """Validate the data structure before attempting insert"""
        if not data:
            raise ValueError("No data provided for insert operation")
        
        if not isinstance(data[0], dict):
            raise ValueError("Data must be a list of dictionaries")
        
        columns = set(data[0].keys())
        for idx, row in enumerate(data[1:], 1):
            if set(row.keys()) != columns:
                raise ValueError(f"Row {idx} has different columns than the first row")

    def _insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        if not data:
            return

        try:
            self._validate_data(data)

            columns = list(data[0].keys())
            placeholders = ",".join(["?" for _ in columns])
            column_names = ",".join(columns)
            
            query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
            
            cursor = self._connection.cursor()
            try:
                for idx, row in enumerate(data):
                    try:
                        values = [row[col] for col in columns]
                        cursor.execute(query, values)
                    except pyodbc.DataError as e:
                        raise SQLServerInsertError(f"Data conversion error in row {idx}: {str(e)}")
                    except pyodbc.Error as e:
                        raise SQLServerInsertError(f"Failed to insert row {idx}: {str(e)}")
                
                self._connection.commit()
            except Exception as e:
                self._connection.rollback()
                raise e
            finally:
                cursor.close()
        except Exception as e:
            logger.error(f"Error during data insertion: {str(e)}")
            raise

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
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return {
                    "error_status": True,
                    "message": f"Invalid JSON data format: {str(e)}. Please ensure the data is valid JSON.",
                }

            if not isinstance(parsed_data, list):
                parsed_data = [parsed_data]

            if not parsed_data:
                return {
                    "error_status": True,
                    "message": "No data provided for insertion",
                }

            try:
                self._create_connection(
                    host=host,
                    port=port,
                    database=database,
                    username=username,
                    password=password,
                )
            except SQLServerAuthenticationError as e:
                return {
                    "error_status": True,
                    "message": f"Authentication failed: {str(e)}. Please check your credentials.",
                }
            except SQLServerConnectionError as e:
                return {
                    "error_status": True,
                    "message": f"Connection error: {str(e)}. Please check your connection settings.",
                }

            try:
                self._insert_data(table_name=table_name, data=parsed_data)
            except SQLServerInsertError as e:
                return {
                    "error_status": True,
                    "message": f"Insert operation failed: {str(e)}",
                }
            except ValueError as e:
                return {
                    "error_status": True,
                    "message": f"Data validation error: {str(e)}",
                }

            return {
                "error_status": False,
                "message": f"Successfully inserted {len(parsed_data)} rows into {table_name}",
            }

        except Exception as e:
            logger.error(f"Unexpected error in SQL Server sink: {str(e)}")
            return {
                "error_status": True,
                "message": f"An unexpected error occurred: {str(e)}",
            }
        finally:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {str(e)}")
                self._connection = None

    def __del__(self):
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                logger.error(f"Error closing connection in destructor: {str(e)}")
