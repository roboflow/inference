Below you can find API reference for **Batch Processing** and **Data Staging** services. Specifications can
be used to build API integrations.

<style>
rapi-doc { background: transparent; }
rapi-doc::part(section-main-content) { background: transparent; }
rapi-doc::part(section-navbar) { background: transparent; }
rapi-doc::part(section-overview) { background: transparent; }
rapi-doc::part(section-tag) { background: transparent; }
rapi-doc::part(section-operations-in-tag) { background: transparent; }
rapi-doc::part(section-operation) { background: transparent; }
rapi-doc::part(section-servers) { background: transparent; }
rapi-doc::part(section-auth) { background: transparent; }
</style>

## Batch Processing API
<rapi-doc spec-url="../batch_processing_swagger.json" render-style="view" theme="light" show-header="false" allow-server-selection="false" bg-color="#ffffff00" primary-color="#6c63ff" regular-font="Inter, sans-serif" mono-font="ui-monospace, monospace" load-fonts="false"></rapi-doc>

## Data Staging API
<rapi-doc spec-url="../data_staging_swagger.json" render-style="view" theme="light" show-header="false" allow-server-selection="false" bg-color="#ffffff00" primary-color="#6c63ff" regular-font="Inter, sans-serif" mono-font="ui-monospace, monospace" load-fonts="false"></rapi-doc>
<script type="module" src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"></script>

### Data Staging batches explained

The **Data Staging** service is responsible for creating and maintaining batches of data—logically organized groups of
files that are either queued for processing or represent the output of a completed processing job.

There are three types of batches:


* **Simple batches** (recognized by `type: simple-batch`) are created when users ingest data one item at a time using
endpoints for individual images or videos. There is no strict size limit, but for best performance, it's recommended to
keep these batches relatively small—ideally up to 5,000–10,000 items.

* **Sharded batches** (recognized by `type: sharded-batch`) are created when users leverage bulk ingestion options and
are currently supported only for images. These batches are designed for large-scale workloads, capable of handling
millions of data points. Data is automatically sharded to support efficient parallel processing.

* **Multipart batches** (recognized by `type: multipart-batch`) are created internally by the system and cannot be
created directly by users. Each multipart batch is a logical grouping of sub-batches (each behaving like a simple or
sharded batch), managed as a single entity. Operations such as listing or exporting are typically performed on
individual parts, which must be listed beforehand.
