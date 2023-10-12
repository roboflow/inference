import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Roboflow Inference Server",
  description: "With no prior knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments using the Roboflow Inference Server.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
        />
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
