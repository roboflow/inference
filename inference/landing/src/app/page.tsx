"use client";

import Image from "next/image";
import HeaderLink from "./components/headerLink";
import React from "react";
import { roboto_mono } from "./fonts";
import ExampleLink from "./components/exampleLink";
import classNames from "classnames";

export default function Home() {
  const [page, setPage] = React.useState("landing");
  return (
    <main className="flex min-h-screen flex-col items-center justify-between gap-0">
      <div
        id="aboveFold"
        className="flex flex-col gap-16 py-16 px-20 items-center w-screen h-screen"
      >
        <div className="flex flex-col items-center gap-6 max-w-xl">
          <div className="flex flex-col gap-1 items-center">
            <img
              src="/roboflow_full_logo_color.png"
              alt="Roboflow Logo"
              width={200}
            />
            <div className="font-bold text-gray-900 text-6xl">Inference</div>
            <div className={roboto_mono.className}>
              developer-friendly vision inference
            </div>
          </div>
          <div className="text-lg font-medium">
            Roboflow Inference is an easy-to-use, production-ready inference
            server for computer vision supporting deployment of many popular
            model architectures and fine-tuned models.
          </div>
          <div className="flex items-center gap-4">
            <HeaderLink
              href="https://docs.roboflow.com/deploy/inference-api-schema"
              className="bg-purple-600"
              label="Documentation"
              icon="file"
              theme="dark"
            />
            <HeaderLink
              href="https://github.com/roboflow/inference"
              className="bg-gray-900"
              label="Github Repository"
              icon="github"
              theme="dark"
            />
          </div>
        </div>

        <div className="flex flex-col gap-2 w-full">
          <div className="text-xl font-semibold text-purple-600 flex gap-4 items-center">
            Example Projects{" "}
            <span
              className={classNames(
                roboto_mono.className,
                "text-base font-medium text-purple-400"
              )}
            >
              Built with Inference
            </span>
          </div>
          <div className="hideScrollbar flex items-center gap-4 w-full overflow-x-auto overflow-y-hidden">
            <ExampleLink
              href="#"
              title="Gaze Detection"
              body="Detects the direction in which someone is looking and the point in a frame at which someone is looking."
              icon="👁️"
            />
            <ExampleLink
              href="#"
              title="Inference Dashboard"
              body="Extract insights from video frames at defined intervals and generates informative visualizations and CSV outputs."
              icon="📊"
            />
            <ExampleLink
              href="#"
              title="Inference Client"
              body="Quickstart HTTP and UDP clients for use with Inference."
              icon="⚡"
            />
          </div>
        </div>
        <div className="flex items-center gap-4">
          <HeaderLink
            href="https://docs.roboflow.com/deploy/inference-api-schema"
            className=""
            label="Start with code snippets"
            icon="file"
            theme="light"
          />
          <HeaderLink
            href="https://github.com/roboflow/inference"
            className=""
            label="Find interesting models"
            icon="search"
            theme="light"
          />
          <HeaderLink
            href="https://github.com/roboflow/inference"
            className=""
            label="Train your own custom models"
            icon="magic"
            theme="light"
          />
        </div>
      </div>
      <div id="dividerGradient" className="h-[3px] w-full">
        {" "}
      </div>
      <div className="px-14 py-8 flex flex-col">
        {page === "landing" && <div>Below the fold</div>}
      </div>
    </main>
  );
}
