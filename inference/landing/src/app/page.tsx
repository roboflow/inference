"use client";

import Image from "next/image";
import HeaderLink from "./components/headerLink";
import React from "react";
import { roboto_mono } from "./fonts";
import ExampleLink from "./components/exampleLink";
import classNames from "classnames";

export default function Home() {
  // Will use when we add links to in progress example page
  // const [page, setPage] = React.useState("landing");
  const [dashboardAvailable, setDashboardAvailable] = React.useState(false);

  React.useEffect(() => {
    fetch("/dashboard.html", { method: "HEAD", cache: "no-store" })
      .then((res) => setDashboardAvailable(res.ok))
      .catch(() => setDashboardAvailable(false));
    return () => {};
  }, []);
  return (
    <main className="flex min-h-screen flex-col items-stretch gap-0">
      <div
        id="aboveFold"
        className="flex flex-col justify-center w-full min-h-screen py-6 md:py-12 overflow-hidden"
      >
        <div className="flex flex-col items-center gap-6 px-8 text-center ">
          <div className="flex flex-col gap-1 items-center relative z-10">
            <a href="https://roboflow.com" target="_blank">
              <Image
                src="/static/roboflow_full_logo_color.svg"
                alt="Roboflow Logo"
                width={200}
                height={100}
              />
            </a>
            <div className="font-bold text-gray-900 text-5xl md:text-6xl">
              Inference
            </div>
            <div
              className={classNames(
                roboto_mono.className,
                "font-bold text-purple-500"
              )}
            >
              developer-friendly vision inference
            </div>
            <Image
              src="/static/cone.svg"
              alt="Roboflow Logo"
              width={120}
              height={120}
              className="hidden md:flex flex-none absolute left-[-400px] -top-20 xl:left-[-450px] xl:-top-28 z-0"
            />
            <Image
              src="/static/trash.svg"
              alt="Roboflow Logo"
              width={110}
              height={110}
              className="hidden md:flex flex-none absolute -left-56 top-4 xl:-left-64 xl:top-6 z-0"
            />
            <Image
              src="/static/boat.svg"
              alt="Roboflow Logo"
              width={100}
              height={100}
              className="hidden md:flex flex-none absolute right-[-375px] top-36 xl:right-[-420px] xl:top-12 z-0"
            />
            <Image
              src="/static/car.svg"
              alt="Roboflow Logo"
              width={140}
              height={140}
              className="hidden md:flex flex-none absolute -right-56 top- xl:-right-56 xl:-top-6 z-0"
            />
          </div>

          <div className="text-base md:text-lg font-medium max-w-xl">
            Roboflow Inference is an{" "}
            <span className="font-extrabold">
              easy-to-use, production-ready
            </span>{" "}
            inference server for computer vision that supports the deployment of
            many popular model architectures and fine-tuned models.
          </div>
          <div className="flex items-center justify-center gap-2 md:gap-4 flex-col sm:flex-row flex-nowrap sm:flex-wrap px-6">
            <HeaderLink
              href="/build"
              className=""
              label="Start building locally"
              icon="💻"
              target="_top"
            />
            <HeaderLink
              href="https://universe.roboflow.com"
              className=""
              label="Find interesting models"
              icon="🔍"
            />
            <HeaderLink
              href="https://roboflow.com/train"
              className=""
              label="Train your own custom models"
              icon="💫"
            />
          </div>
          {/* Dedicated row for Dashboard link with fixed height to avoid layout shift */}
          <div className="w-full flex items-center justify-center px-6 mt-2">
            <div className="h-12 flex items-center justify-center">
              {dashboardAvailable && (
                <HeaderLink
                  href="/dashboard.html"
                  className=""
                  label="View Server Dashboard"
                  icon="📊"
                  target="_top"
                />
              )}
            </div>
          </div>
          <div className="text-base md:text-lg font-medium max-w-xl">
          
          </div>
          <div className="flex w-full items-center justify-center gap-2 flex-col sm:flex-row flex-nowrap sm:flex-wrap sm:px-10">
            <a href="notebook/start" target="_blank"
              className="w-full sm:w-max m-0 lg:mx-12 xl:mx-0 xl:w-max gap-0 xl:gap-8 flex-col lg:flex-row justify-between h-auto lg:h-20 xl:h-16 items-center  group flex bg-[#A351FB]  bg-opacity-10 hover:bg-opacity-20 transition duration-400 shadow-none hover:shadow rounded-sm border border-[#A351FB]"
              
            > <span className="text-[15px] px-6 lg:pl-6 md:pr-0 py-5 md:py-4  whitespace-normal xl:whitespace-nowrap tracking-tight font-semibold text-purple-600 text-left font-mono">This inference server comes with a built in Jupyterlab server for development and testing.</span>
            
            <div className="h-12 w-full lg:h-full lg:w-max font-medium group-hover:bg-purple-600 transition duration-400 bg-[#A351FB] px-5 whitespace-nowrap flex-row text-sm text-white rounded-r-sm flex items-center justify-center">Jump Into an Inference Enabled Notebook    <div className="pl-4 font-bold">→</div></div>
            </a>
          </div>
        </div>
        <div className="mt-10 md:mt-16 flex flex-col w-full self-start">
          <div className="text-xl font-semibold text-gray-900 flex flex-col items-center sm:flex-row gap-1 sm:gap-4 px-4 sm:px-[10%] ">
            Example Projects{" "}
            <div
              className={classNames(
                roboto_mono.className,
                "flex items-center border border-purple-500 rounded px-1.5 py-0.5 text-sm text-purple-600"
              )}
            >
              Built with Inference
            </div>
          </div>
          <div className="hideScrollbar flex items-stretch sm:items-center flex-col sm:flex-row gap-4 w-full overflow-visible sm:overflow-x-auto pt-4 sm:pt-3 pb-12 px-4 sm:px-[10%] ">
            <ExampleLink
              href="https://github.com/roboflow/inference/tree/main/examples/gaze-detection"
              title="Gaze Detection"
              body="Detects the direction in which someone is looking and the point in a frame at which someone is looking."
              icon="👁️"
            />
            <ExampleLink
              href="https://inference.roboflow.com/inference_helpers/inference_sdk/"
              title="Inference Client"
              body="Quickstart HTTP and UDP clients for use with Inference."
              icon="⚡"
            />
            <ExampleLink
              href="https://blog.roboflow.com/how-to-use-computer-vision-to-monitor-inventory/"
              title="Monitor Inventory"
              body="Use computer vision to analyze video streams, extract insights from video frames, and create actionable visualizations and CSV outputs."
              icon="📦"
            />
            <ExampleLink
              href="https://github.com/roboflow/inference/tree/main/examples/inference-dashboard-example"
              title="Create an Inference Dashboard"
              body="Extract insights from video frames at defined intervals and generates informative visualizations and CSV outputs."
              icon="📊"
            />
            <ExampleLink
              href="https://blog.roboflow.com/clip-image-search-faiss/"
              title="Build Image-to-Image Search"
              body="Build a CLIP powered search that uses images as input to find other similar images."
              icon="🖼️"
            />
          </div>
        </div>
        <div className="flex items-center justify-center gap-2 flex-col sm:flex-row">
          {/* TODO: Replace placeholder emoji & → with actual FontAwesome icons from mocks */}
          <HeaderLink
            href="https://inference.roboflow.com"
            className="bg-purple-600"
            label="Read the Documentation"
            icon="📄"
          />
          <HeaderLink
            href="https://github.com/roboflow/inference"
            className="bg-gray-900"
            label="Star the Github Repository"
            icon="⭐"
          />
        </div>
      </div>
      <div id="dividerGradient" className="h-0.5 sm:h-1 w-full">
        {" "}
      </div>
    </main>
  );
}
