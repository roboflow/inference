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
        className="flex flex-col justify-center items-center max-w-full w-full min-h-screen  overflow-hidden"
      >
        <div className="flex flex-col items-center gap-2 md:gap-10 pb-12 md:px-6 lg:px-10 w-full text-center ">
          <div className="flex  pt-12 flex-col gap-1 items-center relative z-10">
            <a href="https://roboflow.com" target="_blank">
              <Image
                src="/static/roboflow_full_logo_color.svg"
                alt="Roboflow Logo"
                width={200}
                height={100}
              />
            </a>
            
            <h1 className="font-bold text-gray-900 text-4xl md:text-6xl">
              Inference
            </h1>
            <h2
              className={classNames(
                roboto_mono.className,
                "font-bold text-base text-purple-500"
              )}
            >
                developer-friendly vision inference
            </h2>
       
          </div>
              <div className="flex w-full xl:w-[1000px] max-w-[1000px] bg-white  py-10 bg-opacity-30 border border-white rounded justify-start items-center flex-col px-3 md:px-6 xl:px-12 gap-2 lg:gap-4">

                <h3 className="px-2 lg:px-0 text-xl md:text-3xl text-left  font-semibold 
                text-gray-900 w-full flex pb-3 ">Jump Into an Inference Enabled Notebook</h3>
                <div className="flex w-full flex-row gap-3 ">
                  <span>•</span>
                  <span className="w-full leading-loose justify-start xl:leading-relaxed text-sm lg:text-base items-baseline gap-0 text-left">
                  To use the built in notebooks in Inference, you need to enable the
                  notebooks feature via the environment variable <span className="font-mono text-xs lg:text-sm font-semibold py-1 px-2 border border-gray-400 text-gray-700 rounded">NOTEBOOK_ENABLED</span> .
                  </span>
                </div>
                <div className="flex py-3 flex-row gap-3 ">
                  <span>•</span>
                  <span className="w-full leading-loose lg:leading-7 text-sm lg:text-base items-baseline gap-0 text-left">
                  To do this, use the <span className="font-mono whitespace-break-spaces break-normal  text-xs lg:text-[13px] bg-black bg-opacity-90 font-normal py-1 px-2 border border-gray-900 mx-1 text-white rounded">--dev</span> flag with the inference-cli: <span className="font-mono break-normal whitespace-break-spaces text-xs lg:text-[13px] bg-black bg-opacity-90 font-normal py-1 px-2 border border-gray-900 mx-1 text-white rounded">inference
            server start --dev</span>. Or, update your docker run command with the
            argument <span className="font-mono whitespace-break-spaces text-xs break-normal lg:text-[13px] bg-black bg-opacity-90 font-normal py-1 px-2 border border-gray-900 mx-1 text-white rounded">-e NOTEBOOK_ENABLED=true -p 9003:9003</span>.</span> 
                 
                </div>
          
          <a
            href="notebook/start"
            className="mt-6 w-max flex flex-row text-white items-center justify-center text-sm lg:text-base rounded py-3 px-8 hover:bg-purple-600 transition duration-400  bg-purple-500 "
            target="_blank"
        >Launch Notebook <div className="pl-2 font-bold">→</div>
          </a>
              </div>
          
        </div>
      </div>
      <div id="dividerGradient" className="h-0.5 sm:h-1 w-full">
        {" "}
      </div>
    </main>
  );
}
