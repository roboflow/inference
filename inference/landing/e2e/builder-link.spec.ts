import { test, expect, type Page, type Route } from "@playwright/test";

/**
 * Regression coverage for the `/build` discovery logic in src/app/page.tsx.
 * These intercept the network layer so they exercise the real client
 * component end to end (real fetch, real React state, real DOM) without
 * needing a running inference backend. Real-server verification against the
 * actual Python backend lives outside this repo, in
 * roboflow/evidence/inference-builder/.
 *
 * Negative controls (documented in roboflow/evidence/inference-builder/EVIDENCE.md,
 * not run automatically here): reverting the `invisible`/always-mounted guard
 * in page.tsx back to `{builderAvailable && (...)}` breaks
 * "no layout shift when the builder becomes available"; changing the probe's
 * method from GET to HEAD breaks "probes /build with GET, not HEAD".
 */

async function mockDashboard(page: Page) {
  // Not under test here; keep it deterministic so it can't add noise.
  await page.route("**/dashboard.html", (route) =>
    route.fulfill({ status: 404, body: "not found" }),
  );
}

function holdableBuildRoute() {
  let release!: (response: { status: number; abort?: boolean }) => void;
  const gate = new Promise<{ status: number; abort?: boolean }>((resolve) => {
    release = resolve;
  });
  let observedMethod: string | undefined;

  const handler = async (route: Route) => {
    observedMethod = route.request().method();
    const outcome = await gate;
    if (outcome.abort) {
      await route.abort("failed");
      return;
    }
    await route.fulfill({
      status: outcome.status,
      contentType: "text/html",
      body: "<html>builder</html>",
    });
  };

  return {
    handler,
    release: (outcome: { status: number; abort?: boolean }) => release(outcome),
    getObservedMethod: () => observedMethod,
  };
}

const findModelsLink = (page: Page) =>
  page.getByRole("link", { name: /Find interesting models/i });
// CSS attribute selector, not getByRole: an accessibility-tree role/name
// query can't reliably find an element while it's `visibility: hidden`
// (such elements are excluded from the accessibility tree entirely), and
// visibility is exactly what these tests assert on.
const builderLink = (page: Page) => page.locator('a[href="/build"]');

test.describe("builder link discovery (/build)", () => {
  test("stays invisible-but-reserved while pending, then appears with no layout shift, when available", async ({
    page,
  }) => {
    await mockDashboard(page);
    const build = holdableBuildRoute();
    await page.route("**/build", build.handler);
    await page.setViewportSize({ width: 390, height: 844 }); // matches the reviewer's repro viewport

    await page.goto("/");
    await expect(builderLink(page)).toHaveCount(1); // always mounted
    await expect(builderLink(page)).not.toBeVisible(); // reserved, not shown, while pending

    const beforeBox = await findModelsLink(page).boundingBox();
    expect(beforeBox).not.toBeNull();

    build.release({ status: 200 });
    await expect(builderLink(page)).toBeVisible();
    await expect(builderLink(page)).toHaveAttribute("href", "/build");

    const afterBox = await findModelsLink(page).boundingBox();
    expect(afterBox).not.toBeNull();
    expect(afterBox!.y).toBeCloseTo(beforeBox!.y, 0);
    expect(afterBox!.x).toBeCloseTo(beforeBox!.x, 0);

    expect(build.getObservedMethod()).toBe("GET");
  });

  test("stays hidden with no layout shift when the backend reports unavailable (404)", async ({
    page,
  }) => {
    await mockDashboard(page);
    const build = holdableBuildRoute();
    await page.route("**/build", build.handler);
    await page.setViewportSize({ width: 390, height: 844 });

    await page.goto("/");
    const beforeBox = await findModelsLink(page).boundingBox();

    build.release({ status: 404 });
    await page.waitForTimeout(200); // let the resolved promise settle
    await expect(builderLink(page)).not.toBeVisible();

    const afterBox = await findModelsLink(page).boundingBox();
    expect(afterBox!.y).toBeCloseTo(beforeBox!.y, 0);
  });

  test("stays hidden and raises no page error when the fetch itself fails", async ({
    page,
  }) => {
    await mockDashboard(page);
    const build = holdableBuildRoute();
    await page.route("**/build", build.handler);

    const pageErrors: Error[] = [];
    page.on("pageerror", (err) => pageErrors.push(err));

    await page.goto("/");
    build.release({ status: 0, abort: true });
    await page.waitForTimeout(200);

    await expect(builderLink(page)).not.toBeVisible();
    expect(pageErrors).toHaveLength(0); // the .catch() handles it; no unhandled rejection surfaces
  });

  test("probes /build with GET, not HEAD", async ({ page }) => {
    await mockDashboard(page);
    let method: string | undefined;
    await page.route("**/build", async (route) => {
      method = route.request().method();
      await route.fulfill({ status: 200, contentType: "text/html", body: "ok" });
    });

    await page.goto("/");
    await expect(builderLink(page)).toBeVisible();
    expect(method).toBe("GET");
  });

  test("aborts a pending /build request when the effect's cleanup runs (development-mode effect-cleanup coverage)", async ({
    page,
  }) => {
    // React StrictMode (dev only) mounts this effect, runs its cleanup, and
    // remounts it once, specifically to surface bugs like a missing
    // AbortController cleanup. That cleanup genuinely fires here and aborts
    // a real in-flight request -- this is real coverage of the cleanup
    // path, not a no-op assertion.
    //
    // This intentionally does NOT reload/navigate to claim broader coverage
    // of "cleanup fires on navigation away": tagging requests by their
    // originating document showed that the extra failure a reload produces
    // belongs to the newly loaded document's own independent StrictMode
    // cycle, not the old document's pending survivor request being aborted
    // by the navigation. Attributing that second failure to the old
    // request was wrong; removed rather than reworded around. Next's dev
    // server has no lighter-weight way to unmount this page's only
    // component without navigating away from it, so a same-document
    // unmount claim isn't tested here either.
    const failures: string[] = [];
    page.on("requestfailed", (request) => {
      if (request.url().endsWith("/build")) {
        failures.push(request.failure()?.errorText ?? "unknown");
      }
    });

    await mockDashboard(page);
    const build = holdableBuildRoute();
    await page.route("**/build", build.handler);

    await page.goto("/");
    await expect
      .poll(() => failures.length, {
        message: "waiting for StrictMode's mount/cleanup/remount cycle to abort the first fetch attempt",
      })
      .toBeGreaterThanOrEqual(1);
  });
});
