# Stream Management

## Overview
This feature is designed to cater to users requiring the execution of inference to generate predictions using 
Roboflow object-detection models, particularly when dealing with online video streams. 
It enhances the functionalities of the familiar `inference.Stream()` and InferencePipeline() interfaces, as found in 
the open-source version of the library, by introducing a sophisticated management layer. The inclusion of additional 
capabilities empowers users to remotely manage the state of inference pipelines through the HTTP management interface 
integrated into this package.

This functionality proves beneficial in various scenarios, **including but not limited to**:

* Performing inference across multiple online video streams simultaneously.
* Executing inference on multiple devices that necessitate coordination.
* Establishing a monitoring layer to oversee video processing based on the `inference` package.


## Design
![Stream Management - design](./assets/stream_management_api_design.jpg)
