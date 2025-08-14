"use client";

import React, { useEffect, useState } from "react";
import { roboto_mono } from "../fonts";
import classNames from "classnames";

interface ServerInfo {
  name: string;
  version: string;
  uuid: string;
}

interface ModelInfo {
  model_id: string;
  task_type: string;
  batch_size: number;
}

interface ModelsResponse {
  models: ModelInfo[];
}

export default function Dashboard() {
  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [healthStatus, setHealthStatus] = useState<"healthy" | "error" | "loading">("loading");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch server info and use it for health status too
        const infoResponse = await fetch("/info");
        if (infoResponse.ok) {
          const info = await infoResponse.json();
          setServerInfo(info);
          setHealthStatus("healthy"); // If /info works, server is healthy
        } else {
          setHealthStatus("error");
        }

        // Fetch models
        const modelsResponse = await fetch("/model/registry");
        if (modelsResponse.ok) {
          const modelsData: ModelsResponse = await modelsResponse.json();
          setModels(modelsData.models || []);
        }
      } catch (error) {
        console.error("Error fetching dashboard data:", error);
        setHealthStatus("error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    // Refresh data every 10 seconds
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center p-8">
        <div className="text-xl">Loading dashboard...</div>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto w-full">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Inference Dashboard
          </h1>
          <p className="text-gray-600">
            Monitor your Roboflow Inference server status and metrics
          </p>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Server Status Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">
                Server Status
              </h2>
              <div
                className={classNames(
                  "px-3 py-1 rounded-full text-sm font-medium",
                  {
                    "bg-green-100 text-green-800": healthStatus === "healthy",
                    "bg-red-100 text-red-800": healthStatus === "error",
                    "bg-gray-100 text-gray-800": healthStatus === "loading",
                  }
                )}
              >
                {healthStatus}
              </div>
            </div>
            {serverInfo && (
              <div className="space-y-2">
                <div>
                  <span className="font-medium text-gray-700">Version: </span>
                  <span className={roboto_mono.className}>{serverInfo.version}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Server ID: </span>
                  <span className={classNames(roboto_mono.className, "text-sm")}>
                    {serverInfo.uuid}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Models Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">
                Loaded Models
              </h2>
              <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                {models.length}
              </div>
            </div>
            <div className="space-y-2">
              {models.length === 0 ? (
                <p className="text-gray-500 text-sm">No models loaded</p>
              ) : (
                <div className="max-h-48 overflow-y-auto space-y-2">
                  {models.map((model, index) => (
                    <div
                      key={index}
                      className="border border-gray-200 rounded p-3 text-sm"
                    >
                      <div className="font-medium text-gray-900">
                        {model.model_id}
                      </div>
                      <div className="text-gray-600">
                        Task: {model.task_type} | Batch Size: {model.batch_size}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* System Info Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              System Information
            </h2>
            <div className="space-y-2">
              <div>
                <span className="font-medium text-gray-700">Server Port: </span>
                <span className={roboto_mono.className}>9001</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Environment: </span>
                <span className={roboto_mono.className}>Development</span>
              </div>
              {serverInfo && (
                <div>
                  <span className="font-medium text-gray-700">Server Name: </span>
                  <span className={roboto_mono.className}>{serverInfo.name}</span>
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions Card */}
          <div className="bg-white rounded-lg shadow-md p-6 md:col-span-2 lg:col-span-3">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Quick Actions
            </h2>
            <div className="flex flex-wrap gap-4">
              <a
                href="/notebook/start"
                target="_blank"
                className="inline-flex items-center px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors"
              >
                Open Jupyter Notebook
              </a>
              <a
                href="/"
                className="inline-flex items-center px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
              >
                Back to Home
              </a>
              <button
                onClick={() => window.location.reload()}
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Refresh Data
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}