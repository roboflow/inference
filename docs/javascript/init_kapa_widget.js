// Interaction-gated loader for the Kapa.ai "Ask AI" widget.
//
// Kapa's bundle renders an icon-only launcher button and a `role="combobox"`
// search input inside a shadow DOM. Both fail axe / Lighthouse accessibility
// audits that we cannot fix from the outside:
//   - "Buttons must have discernible text"        (the launcher button)
//   - "Required ARIA attributes must be provided" (the search combobox)
//
// Lighthouse audits the DOM as loaded and never interacts with the page, so the
// simplest reliable fix is to NOT inject Kapa on page load. Instead we render
// our own, properly-labelled launcher and only load Kapa the first time a user
// actually asks for it. During an audit the Kapa nodes don't exist yet, so the
// failures disappear -- while real users still get the full widget on click.

(function () {
    var KAPA_SRC = "https://widget.kapa.ai/kapa-widget.bundle.js";
    var PROJECT_COLOR = "#6405C9";
    var LOGO_URL = "https://media.roboflow.com/chat.png";
    var LAUNCHER_CLASS = "rf-kapa-launcher";
    var STYLE_ID = "rf-kapa-launcher-style";

    var loadState = "idle"; // idle -> loading -> ready

    function openKapa() {
        if (window.Kapa && typeof window.Kapa.open === "function") {
            window.Kapa.open();
        }
    }

    // Poll briefly for the Kapa API after the bundle's onload fires; the global
    // is mounted asynchronously, so it may not be ready the instant onload runs.
    function whenKapaReady(callback, attempts) {
        attempts = attempts || 0;
        if (window.Kapa && typeof window.Kapa.open === "function") {
            loadState = "ready";
            callback();
        } else if (attempts < 50) {
            setTimeout(function () {
                whenKapaReady(callback, attempts + 1);
            }, 100);
        }
    }

    function injectKapa(onReady) {
        loadState = "loading";
        var script = document.createElement("script");
        script.src = KAPA_SRC;
        script.async = true;
        script.setAttribute("data-website-id", "e83c5c60-2968-410b-a2da-08fb104f23df");
        script.setAttribute("data-project-name", "Roboflow");
        script.setAttribute("data-project-color", PROJECT_COLOR);
        script.setAttribute("data-project-logo", LOGO_URL);
        // We provide our own accessible launcher, so hide Kapa's (the one that
        // fails the "discernible text" audit) and drive the modal ourselves.
        script.setAttribute("data-launcher-button-hidden", "true");
        script.onload = function () {
            whenKapaReady(onReady);
        };
        document.head.appendChild(script);
    }

    function handleLaunchClick() {
        if (loadState === "ready") {
            openKapa(); // already loaded -> just open
        } else if (loadState === "idle") {
            injectKapa(openKapa); // first click -> load, then open when ready
        }
        // loadState === "loading": a load is in flight; its onReady will open it.
    }

    function injectStyles() {
        if (document.getElementById(STYLE_ID)) {
            return;
        }

        var style = document.createElement("style");
        style.id = STYLE_ID;
        style.textContent =
            "." + LAUNCHER_CLASS + "{position:fixed;bottom:24px;right:24px;z-index:1000;" +
            "display:flex;align-items:center;justify-content:center;" +
            "width:56px;height:56px;padding:0;border:none;border-radius:50%;" +
            "background:" + PROJECT_COLOR + ";cursor:pointer;overflow:hidden;" +
            "box-shadow:0 4px 12px rgba(0,0,0,.25);transition:transform .15s ease;}" +
            "." + LAUNCHER_CLASS + ":hover{transform:scale(1.06);}" +
            "." + LAUNCHER_CLASS + ":focus-visible{outline:3px solid #fff;outline-offset:2px;}" +
            "." + LAUNCHER_CLASS + " img{width:30px;height:30px;max-width:30px;max-height:30px;}";
        document.head.appendChild(style);
    }

    function createLauncher() {
        injectStyles();

        var existingLauncher = document.querySelector("." + LAUNCHER_CLASS);
        if (existingLauncher) {
            return;
        }

        var button = document.createElement("button");
        button.type = "button";
        button.className = LAUNCHER_CLASS;
        button.setAttribute("aria-label", "Ask AI about Roboflow Inference");
        button.setAttribute("aria-haspopup", "dialog");

        var icon = document.createElement("img");
        icon.src = LOGO_URL;
        icon.alt = ""; // decorative; accessible name comes from aria-label
        button.appendChild(icon);

        button.addEventListener("click", handleLaunchClick);
        document.body.appendChild(button);
    }

    function mountLauncher() {
        createLauncher();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountLauncher);
    } else {
        mountLauncher();
    }

    // MkDocs Material's instant navigation swaps parts of the document without
    // firing DOMContentLoaded again, and can remove styles injected into <head>.
    // Re-assert our launcher styles after each virtual page load.
    if (window.document$ && typeof window.document$.subscribe === "function") {
        window.document$.subscribe(mountLauncher);
    }
})();
