# Processing data with Workflows

`inference workflows` command provides a way to process images and videos with Workflow. It is possible to 
process:

* individual images

* directories of images

* video files


!!! Tip "Discovering command capabilities"

    To check detail of the command, run:
    
    ```bash
    inference workflows --help
    ```

    Additionally, help guide is also available for each sub-command:

    ```bash
    inference workflows process-image --help
    ```


## Process individual image

Basic usage of the command is illustrated below:

```bash
inference workflows process-image \
    --image_path {your-input-image} \
    --output_dir {your-output-dir} \
    --workspace_name {your-roboflow-workspace-url} \
    --workflow_id {your-workflow-id} \
    --api-key {your_roboflow_api_key}
```

which would take your input image, run it against your Workflow and save results in output directory. By default, 
Workflow will be processed using Roboflow Hosted API. You can tweak behaviour of the command:

* if you want to process the image locally, using `inference` Python package - use 
`--processing_target inference_package` option (*requires `inference` to be installed*)

* to see all options, use `inference workflows process-image --help` command

* any option that starts from `--` which is not enlisted in `--help` command will be treated as input parameter 
to the workflow execution - with automatic type conversion applied. Additionally `--workflow_params` option may 
specify path to `*.json` file providing workflow parameters (explicit parameters will override parameters defined 
in file)

* if your Workflow defines image parameter placeholder under a name different from `image`, you can point the 
proper image input by `--image_input_name`

* `--allow_override` flag must be used if output directory is not empty


## Process directory of images

Basic usage of the command is illustrated below:

```bash
inference workflows process-images-directory \
    -i {your_input_directory} \
    -o {your_output_directory} \
    --workspace_name {your-roboflow-workspace-url} \
    --workflow_id {your-workflow-id} \
    --api-key {your_roboflow_api_key}
```

You can tweak behaviour of the command:

* if you want to process the image locally, using `inference` Python package - use 
`--processing_target inference_package` option (*requires `inference` to be installed*)

* to see all options, use `inference workflows process-image --help` command

* any option that starts from `--` which is not enlisted in `--help` command will be treated as input parameter 
to the workflow execution - with automatic type conversion applied. Additionally `--workflow_params` option may 
specify path to `*.json` file providing workflow parameters (explicit parameters will override parameters defined 
in file)

* if your Workflow defines image parameter placeholder under a name different from `image`, you can point the 
proper image input by `--image_input_name`

* `--allow_override` flag must be used if output directory is not empty

* `--threads` option can specify number of threads used to run the requests when processing target is API

## Process video file 

!!! Note "`inference` required"

    This command requires `inference` to be installed.

Basic usage of the command is illustrated below:

```bash
inference workflows process-video \
    --video_path {video_to_be_processed} \
    --output_dir {empty_directory} \
    --workspace_name {your-roboflow-workspace-url} \
    --workflow_id {your-workflow-id} \
    --api-key {your_roboflow_api_key}
```

You can tweak behaviour of the command:

* `--max_fps` option can be used to subsample video frames while processing

* to see all options, use `inference workflows process-image --help` command

* any option that starts from `--` which is not enlisted in `--help` command will be treated as input parameter 
to the workflow execution - with automatic type conversion applied. Additionally `--workflow_params` option may 
specify path to `*.json` file providing workflow parameters (explicit parameters will override parameters defined 
in file)

* if your Workflow defines image parameter placeholder under a name different from `image`, you can point the 
proper image input by `--image_input_name`

* `--allow_override` flag must be used if output directory is not empty
