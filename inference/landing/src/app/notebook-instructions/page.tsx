"use client";

import Image from "next/image";
import HeaderLink from "@/app/components/headerLink";
import React from "react";
import { roboto_mono } from "@/app/fonts";
import ExampleLink from "@/app/components/exampleLink";
import classNames from "classnames";

export default function Home() {
  // Will use when we add links to in progress example page
  // const [page, setPage] = React.useState("landing");
  return (
    <main className="flex min-h-screen flex-col items-stretch gap-0">
      <div
        id="aboveFold"
        className="flex flex-col justify-center w-full min-h-screen py-6 md:py-12 overflow-hidden"
      >
        <div className="flex flex-col items-center gap-6 px-8 text-center ">
          <div className="flex flex-col gap-1 items-center relative z-10">
            <a href="https://roboflow.com" target="_blank">
              <img
                src="/roboflow_full_logo_color.svg"
                alt="Roboflow Logo"
                width={200}
              />
            </a>
            <div className="font-bold text-gray-900 text-5xl md:text-6xl">
              Inference Notebook
            </div>
            <div
              className={classNames(
                roboto_mono.className,
                "font-bold text-purple-500"
              )}
            >
              jump into an inference enabled notebook
            </div>
          </div>

          <div className="flex items-center justify-center gap-2 flex-col sm:flex-row flex-nowrap sm:flex-wrap px-6">
            To use the built in notebooks in Inference, you need to enable the
            notebooks feature via the environment variable NOTEBOOK_ENABLED.
          </div>
          <div className="flex items-center justify-center gap-2 flex-col sm:flex-row flex-nowrap sm:flex-wrap px-6">
            To do this, use the `--dev` flag with the inference-cli: `inference
            server start --dev`. Or, update your docker run command with the
            argument `-e NOTEBOOK_ENABLED=true`
          </div>
          <HeaderLink
            href="http://localhost:9001/notebook/start"
            className=""
            label="Launch Notebook"
            icon="📓"
          />
        </div>
      </div>
      <div id="dividerGradient" className="h-0.5 sm:h-1 w-full">
        {" "}
      </div>
    </main>
  );
}
