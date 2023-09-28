import Image from "next/image";
import HeaderLink from "./components/headerLink";
import React from "react";

export default function Home() {
  const [page, setPage] = React.useState("landing");
  return (
    <main className="flex min-h-screen flex-col items-center justify-between gap-0">
      <div className="">Above the fold</div>
      {page === "snippets" && <div>Below the fold</div>}
    </main>
  );
}
