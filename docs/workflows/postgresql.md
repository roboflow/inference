# Write workflow results to PostgreSQL

The Inference runtime includes `psycopg[binary]`. Custom Python code executing in
that runtime can `import psycopg` without installing a package during execution.
A remote Custom Python environment must also use an image containing the driver.
Existing workers must adopt the new runtime image before it becomes available.

The **PostgreSQL Sink** block replaces customer-written connection and insert code.
Enable enterprise blocks with `LOAD_ENTERPRISE_BLOCKS=True`. Hosted serverless execution is not supported; cloud streaming
workers must explicitly load the block and have a network path to the database.

## Example

Create the destination table before running the workflow:

```sql
CREATE TABLE public.detections (
    camera_id text NOT NULL,
    label text NOT NULL,
    confidence double precision
);
```

Use the following workflow, supplying `database_host`, `database_password`, and
`row` at execution time through the runtime's supported input/secret delivery path.
Do not save a real password as a literal in a shared workflow.

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowParameter", "name": "database_host"},
    {"type": "WorkflowParameter", "name": "database_password"},
    {"type": "WorkflowParameter", "name": "row"}
  ],
  "steps": [{
    "type": "roboflow_core/postgresql_sink@v1",
    "name": "save_detection",
    "host": "$inputs.database_host",
    "port": 5432,
    "database": "production",
    "username": "workflow_writer",
    "password": "$inputs.database_password",
    "schema_name": "public",
    "table_name": "detections",
    "data": "$inputs.row",
    "sslmode": "verify-full",
    "fire_and_forget": false
  }],
  "outputs": [
    {"type": "JsonField", "name": "error_status", "selector": "$steps.save_detection.error_status"},
    {"type": "JsonField", "name": "message", "selector": "$steps.save_detection.message"}
  ]
}
```

For example, `row` can be `{"camera_id": "line-1", "label": "part", "confidence": 0.95}`.
In an inference workflow, wire `data` to an upstream dictionary output instead.
A list of rows must be non-empty and every row must have the same column names.
Column order within a dictionary does not matter. Values retain their native driver
representation; `None` becomes SQL NULL. Nested objects are not automatically
converted to JSON.

## Connection and execution behavior

- The writer needs schema access, INSERT permission, and sequence permissions when
  the table uses a sequence. The block does not create tables or perform upserts.
- Schema, table and column names are raw, case-sensitive identifiers; do not include
  SQL quotes. Specify the schema separately rather than putting `schema.table` in
  `table_name`.
- Configure libpq's trusted root certificate for `verify-full`. The default
  `sslmode=require` requires encryption but does not guarantee hostname verification.
  An isolated local database without TLS needs explicit `sslmode=disable`.
- `connect_timeout` defaults to 10 seconds. `statement_timeout` defaults to 10000
  milliseconds per server statement, including lock waits. These are not a total
  network deadline.
- Synchronous execution reports success only after the complete batch commits.
  Insertion errors roll back the batch. The block does not retry; a connection loss
  during commit may leave the outcome unknown.
- `fire_and_forget` defaults to true for parity with Microsoft SQL Server Sink.
  Its success response means **scheduled**, not **committed**. Errors are logged;
  queued work can be lost on shutdown. Prefer false when streaming to observe
  persistence failures and avoid accumulating background writes to a slow database.
- Workflow sink-disabling policy prevents any scheduling or database connection.

For cloud streaming, deploy a compatible worker image and configure database
reachability and credentials before using this example. This block does not
provision private routes or alter database firewall rules.
