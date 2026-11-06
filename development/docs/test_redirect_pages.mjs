// Run after building the site: node --test development/docs/test_redirect_pages.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

const homepage = readFileSync("site/index.html", "utf8");
const installPage = readFileSync("site/install/index.html", "utf8");
const destination = "https://docs.roboflow.com/deployment/self-hosted/self-hosted";

function redirectFrom(html, suffix) {
    const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/gi)];
    assert.equal(scripts.length, 1, "The standalone redirect has one inline script");
    const source = new URL(`https://inference.roboflow.com/${suffix}`);
    const location = { search: source.search, hash: source.hash, href: source.href };
    runInNewContext(scripts[0][1], { window: { location }, location, URL, URLSearchParams });
    return String(location.href);
}

test("homepage keeps the review's campaign parameters before the anchor", () => {
    assert.equal(
        redirectFrom(homepage, "?utm_source=partner&utm_campaign=launch#install"),
        `${destination}?utm_source=partner&utm_campaign=launch#install`,
    );
});

test("homepage keeps all parameters supported by Roboflow's utm.js", () => {
    const query = "?utm_source=partner&utm_medium=email&utm_campaign=launch&ref=friend&fbclid=click123";
    assert.equal(redirectFrom(homepage, query), destination + query);
});

test("homepage does not forward unrelated query parameters", () => {
    assert.equal(
        redirectFrom(homepage, "?token=private&next=https://example.com&utm_source=partner#install"),
        `${destination}?utm_source=partner#install`,
    );
});

test("campaign values keep encoded characters without becoming URL structure", () => {
    const actual = new URL(redirectFrom(homepage, "?utm_source=partner%26admin%3Dtrue&utm_campaign=a%2Bb%23c%20d#install"));
    assert.equal(actual.searchParams.get("utm_source"), "partner&admin=true");
    assert.equal(actual.searchParams.get("utm_campaign"), "a+b#c d");
    assert.equal(actual.searchParams.has("admin"), false);
    assert.equal(actual.hash, "#install");
});

test("homepage preserves empty values and uses the first repeated campaign value", () => {
    assert.equal(
        redirectFrom(homepage, "?utm_source=first&utm_source=second&utm_campaign=#install"),
        `${destination}?utm_source=first&utm_campaign=#install`,
    );
});

test("homepage still redirects plain and anchor-only visits", () => {
    assert.equal(redirectFrom(homepage, ""), destination);
    assert.equal(redirectFrom(homepage, "#install"), `${destination}#install`);
    assert.equal(redirectFrom(homepage, "?unrelated=value"), destination);
});

test("per-page redirects retain their destination and existing query behavior", () => {
    assert.equal(
        redirectFrom(installPage, "?utm_source=partner#docker"),
        "https://docs.roboflow.com/deployment/self-hosted/inference-server/install#docker",
    );
    assert.match(installPage, /<meta name="robots" content="noindex,follow">/);
});

test("script extraction accepts case-insensitive HTML tags", () => {
    const uppercaseTags = homepage
        .replaceAll("<script>", "<SCRIPT>")
        .replaceAll("</script>", "</SCRIPT>");
    assert.equal(
        redirectFrom(uppercaseTags, "?utm_source=partner&utm_campaign=launch#install"),
        `${destination}?utm_source=partner&utm_campaign=launch#install`,
    );
});
