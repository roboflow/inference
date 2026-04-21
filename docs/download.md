# Downloading Roboflow Inference

<div id="download-status" style="text-align: center; margin: 2rem 0;" >
    <h1>Thanks for Downloading Roboflow Inference!</h1>
    <p id="docs-link"><a href="/install/">Getting Started with Roboflow Inference</a></p>
    <p id="download-message">Your download should start automatically. If it doesn't, <a id="manual-download-link" href="/install">click here to download manually</a>.</p>
</div>

<script>
(function() {

    function getOperatingSystem() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;

        if (userAgent.indexOf('Win') !== -1) return 'windows';
        if (userAgent.indexOf('Mac') !== -1) return 'mac';
        return 'other';
    }

    function pickAssetForOS(assets, os) {
        if (os === 'windows') {
            return assets.find(a => /\.exe$/i.test(a.name) && /installer/i.test(a.name));
        } else if (os === 'mac') {
            return assets.find(a => /\.dmg$/i.test(a.name));
        }
        return null;
    }

    function triggerDownload(downloadURL) {
        const downloadLink = document.createElement('a');
        downloadLink.href = downloadURL;
        downloadLink.download = '';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }

    function showFallback(message) {
        const msgEl = document.getElementById('download-message');
        if (msgEl && message) {
            msgEl.textContent = message;
        }
    }

    async function startDownload() {
        const os = getOperatingSystem();

        if (os === 'other') {
            window.location.href = '/install/';
            return;
        }

        try {
            const res = await fetch('https://api.github.com/repos/roboflow/inference/releases/latest');
            if (!res.ok) throw new Error(`GitHub API returned ${res.status}`);
            const release = await res.json();
            const asset = pickAssetForOS(release.assets || [], os);

            if (!asset) {
                showFallback(
                    os === 'mac'
                        ? "No macOS build is available for the latest release. Please see the install docs for alternatives."
                        : "No installer found for your platform in the latest release. Please download manually from the releases page."
                );
                const manualLink = document.getElementById('manual-download-link');
                if (manualLink) manualLink.href = 'https://github.com/roboflow/inference/releases/latest';
                return;
            }

            const manualLink = document.getElementById('manual-download-link');
            if (manualLink) manualLink.href = asset.browser_download_url;

            triggerDownload(asset.browser_download_url);
        } catch (err) {
            showFallback("We couldn't resolve the latest release automatically. Please download manually from the releases page.");
            const manualLink = document.getElementById('manual-download-link');
            if (manualLink) manualLink.href = 'https://github.com/roboflow/inference/releases/latest';
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startDownload);
    } else {
        startDownload();
    }
})();
</script>

<style>
#download-status {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 2rem;
    margin: 2rem auto;
    max-width: 600px;
}

#download-status p {
    margin: 0.5rem 0;
}

#docs-link {
    margin-top: 1rem;
    margin-bottom: 1rem;
    font-size: 1.2em;
}
</style>