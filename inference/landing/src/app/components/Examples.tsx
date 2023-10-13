"use client";

import React from "react"
import { Tab } from '@headlessui/react'

import { Light as SyntaxHighlighter } from "react-syntax-highlighter"
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python"
import docco from "react-syntax-highlighter/dist/esm/styles/hljs/docco"

SyntaxHighlighter.registerLanguage("python", python);

export default function Examples() {
    return (
        <div className="text-gray-800">
            <Quickstart />
        </div>
    )
}

const dockerCode = `import requests

dataset_id = "soccer-players-5fuqs"
version_id = "1"
image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"
# Replace ROBOFLOW_API_KEY with your Roboflow API Key
api_key = "ROBOFLOW_API_KEY"
confidence = 0.5

url = f"http://localhost:9001/{dataset_id}/{version_id}"

params = {
    "api_key": api_key,
    "confidence": confidence,
    "image": image_url,
}

res = requests.post(url, params=params)
print(res.json())
`

const pipCode = `from inference.models.utils import get_roboflow_model

model = get_roboflow_model(
    model_id="soccer-players-5fuqs/1",
    # Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key="ROBOFLOW_API_KEY"
)

results = model.infer(
    image="https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg",
    confidence=0.5,
    iou_threshold=0.5
)

print(results)
`

const clipCode = `from inference.models import Clip

model = Clip(
    # Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key = "ROBOFLOW_API_KEY"
)

image_url = "https://source.roboflow.com/7fLqS2r1SV8mm0YzyI0c/yy6hjtPUFFkq4yAvhkvs/original.jpg"

embeddings = model.embed_image(image_url)

print(embeddings)
`

const samCode = `from inference.models import SegmentAnything

model = SegmentAnything(
    # Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key = "ROBOFLOW_API_KEY"
)

image_url = "https://source.roboflow.com/7fLqS2r1SV8mm0YzyI0c/yy6hjtPUFFkq4yAvhkvs/original.jpg"

embeddings = model.embed_image(image_url)

print(embeddings)
`

const tabClasses = "py-3 px-4 text-sm ui-selected:border-b-2 border-b ui-selected:border-purple-600 border-white focus:outline-none ui-selected:text-purple-600 text-gray-800 hover:text-purple-600"

function Quickstart() {
    return (
        <div className="w-100">
            <Tab.Group>
                <Tab.List className="flex text-gray-800 mb-3 font-medium">
                    <Tab className={tabClasses}>Docker</Tab>
                    <Tab className={tabClasses}>CLIP</Tab>
                    <Tab className={tabClasses}>SAM</Tab>
                    <Tab className={tabClasses}>pip</Tab>
                </Tab.List>
                <Tab.Panels>
                    <Tab.Panel>
                        <Code code={dockerCode} />
                    </Tab.Panel>
                    <Tab.Panel>
                        <Code code={clipCode} />
                    </Tab.Panel>
                    <Tab.Panel>
                        <Code code={samCode} />
                    </Tab.Panel>
                    <Tab.Panel>
                        <Code code={pipCode} />
                    </Tab.Panel>
                </Tab.Panels>
            </Tab.Group>
        </div>
    )
}


type CodeProps = {
    code: string
}

function Code({ code }: CodeProps) {
    function copyToClipboard() {
        navigator.clipboard.writeText(code)
    }

    return (
        <div className="rounded p-8 bg-[#f8f8ff] relative">
            <SyntaxHighlighter className="rounded p-8" language="python" style={docco}>
                {code}
            </SyntaxHighlighter>
            <div
                className="text-gray-700 absolute w-24 h-8 top-2 right-2 bg-white text-sm flex items-center justify-center border rounded-tr-md rounded-bl-md cursor-pointer select-none hover:bg-gray-100"
                onClick={copyToClipboard}
            >
                <i className="far fa-copy h-4 w-4 mr-1 text-sm" />
                Copy
            </div>
        </div>
    )
}
