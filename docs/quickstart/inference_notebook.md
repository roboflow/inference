Roboflow Inference Servers come equipped with a built in Jupyterlab environment. This environment is the fastest way to get up and running with inference for development and testing. To use it, first start an inference server.

The easiest way to start an inference server is with the inference CLI. Install it via pip:

```bash
pip install inference-cli
```

Now run the `inference sever start` command. Be sure to specify the `--dev` flag so that the notebook environment is enabled (it is disabled by default).

```
inference server start --dev
```

Now visit <a href="http://localhost:9001" target="_blank">localhost:9001</a> in your browser to see the `inference` landing page. This page contains links to resources and examples related to `inference`. It also contains a link to the built in Jupyterlab environment.

<div style="text-align: center;">
<img src="https://storage.googleapis.com/com-roboflow-marketing/inference/inference_landing_page.png" alt="Inference Landing Page" width="400"/>
</div>


From the landing page, select the button labeled "Jump Into an Inference Enabled Notebook" to open a new tab for the Jupyterlab environment. 

<div style="text-align: center;">
<img src="https://storage.googleapis.com/com-roboflow-marketing/inference/inference_jupyterlab_link.png" alt="Inference Jupyterlab Link" width="300"/>
</div>

This Jupyterlab environment comes preloaded with several example notebooks and all of the dependencies needed to run `inference`.

<div style="text-align: center;">
<img src="https://storage.googleapis.com/com-roboflow-marketing/inference/inference_jupyter_lab_quickstart.png" alt="Inference Jupyterlab Link" width="300"/>
</div>
