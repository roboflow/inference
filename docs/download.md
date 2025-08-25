# Downloading Roboflow Inference

<div id="download-status" style="text-align: center; margin: 2rem 0;" >
    <h1>Thanks for Downloading Roboflow Inference!</h1>
    <p id="docs-link"><a href="/install/">Getting Started with Roboflow Inference</a></p>
    <p id="download-message">Your download should start automatically. If it doesn't, <a id="manual-download-link" href="/install">click here to download manually</a>.</p>
</div>

<script>
(function() {
    
    // Cache DOM elements
    const elements = {
        manualLink: null
    };
    
    function getOperatingSystem() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        
        if (userAgent.indexOf('Win') !== -1) return 'windows';
        if (userAgent.indexOf('Mac') !== -1) return 'mac';
        return 'other';
    }
    
    function getDownloadURL(os, version) {
        if(os === 'windows'){
            return `https://github.com/roboflow/inference/releases/download/v${version}/inference-${version}-installer.exe`;
        }else if( os === 'mac'){
            return `https://github.com/roboflow/inference/releases/download/v${version}/Roboflow-Inference-${version}.dmg`;
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
    
    function startDownload() {
        const os = getOperatingSystem();
        const version = '{{ VERSION }}';
        
        // Cache DOM elements on first use
        if (!elements.manualLink) {
            elements.manualLink = document.getElementById('manual-download-link');
        }
        
        if (os === 'other') {
            window.location.href = '/install/setup';
            return;
        }
        
        const downloadURL = getDownloadURL(os, version);
        if (!downloadURL) return;
        
        // Start automatic download
        triggerDownload(downloadURL);
    }
    
    // Start download when page loads
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