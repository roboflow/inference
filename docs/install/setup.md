# Install Inference

Inference is designed to run both on the edge and in the cloud. When used as a server, the primary interface is HTTP for running models and Workflows. You can install and run it in several ways: Docker images for many CPU/GPU platforms, native desktop apps for Windows and macOS, or as a Python package to run locally. Choose the option below that best fits your environment for step-by-step instructions.


<h2 id="native">Windows and macOS Native Installers</h2>

You can now run Roboflow Inference Server on your Windows or macOS machine with our native desktop applications! This is the quickest and most effortless way to get up and running.

<div class="download-container">
    <div class="download-card">
        <a href="https://github.com/roboflow/inference/releases/download/v{{ VERSION }}/inference-{{ VERSION }}-installer.exe" class="download-button">
            <img src="/images/windows-icon.svg" alt="Windows" /> Download for Windows
        </a>
    </div>
    
    <div class="download-card">
        <a href="https://github.com/roboflow/inference/releases/download/v{{ VERSION }}/Roboflow-Inference-{{ VERSION }}.dmg" class="download-button">
            <img src="/images/macos-icon.svg" alt="macOS" /> Download for Mac
        </a>
    </div>
</div>

<p style="text-align: center; font-size: 0.9em; margin-top: 0.5rem;">
    <a href="https://github.com/roboflow/inference/releases" >I need a previous release</a>
</p>

### Windows (x86)
 - [Download the latest installer](https://github.com/roboflow/inference/releases/download/v{{ VERSION }}/inference-{{ VERSION }}-installer.exe) and run it to install Roboflow Inference
 - When the install is finished it will offer to launch the Inference server after the setup completes
 - To stop the inference server simply close the terminal window it opens
 - To start it again later, you can find Roboflow Inference in your Start Menu

### MacOS (Apple Silicon)
 - [Download the Roboflow Inference DMG](https://github.com/roboflow/inference/releases/download/v{{ VERSION }}/Roboflow-Inference-{{ VERSION }}.dmg) 
 - Mount the DMG by double clicking it
 - Drag the Roboflow Inference App to the Application Folder
 - Go to your Application Folder and double click the Roboflow Inference App to start the server


## Run on the Edge

Inference runs on many edge and personal computing devices. Choose your device below to find the installation guide you need:

- [Windows](install/windows.md)
- [macOS](install/mac.md)
- [NVIDIA Jetson](install/jetson.md)
- [Raspberry Pi](install/jetson.md)

## Run in the Cloud

You can run Inference on servers in the cloud. Choose your cloud below to find the installation guide you need:

- [Amazon Web Services](install/cloud/aws.md)
- [Microsoft Azure](install/cloud/azure.md)
- [Google Cloud Platform](install/cloud/gcp.md)

## Run with Roboflow

You can run Workflows developed with Inference in the Roboflow Cloud. You can use:

- [Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments), cloud CPUs or GPUs dedicated to your Workflows.
- [Serverless API](https://docs.roboflow.com/deploy/serverless), which auto-scales with your workloads.