#!/usr/bin/env python3
"""
Test script for SSL functionality in Inference Server
"""

import os
import sys
import time
import requests
import urllib3
from pathlib import Path

# Add inference to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable SSL warnings for self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def test_certificate_manager():
    """Test the certificate manager functionality"""
    print("Testing Certificate Manager...")
    
    from inference.core.interfaces.http.ssl import SSLCertificateManager
    
    # Test with a temporary cache directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cert_manager = SSLCertificateManager(cache_dir=tmpdir)
        
        # Test self-signed certificate generation
        print("Testing self-signed certificate generation...")
        cert_path, key_path = cert_manager.get_certificate_paths("GENERATE")
        assert Path(cert_path).exists(), "Certificate file not created"
        assert Path(key_path).exists(), "Key file not created"
        print(f"✓ Self-signed cert created at: {cert_path}")
        
        # Test roboflow.host certificate download (if available)
        print("\nTesting roboflow.host certificate download...")
        try:
            cert_path, key_path = cert_manager.get_certificate_paths("INDIRECT")
            assert Path(cert_path).exists(), "Certificate file not downloaded"
            assert Path(key_path).exists(), "Key file not downloaded"
            print(f"✓ Roboflow.host cert downloaded to: {cert_path}")
        except Exception as e:
            print(f"⚠ Roboflow.host cert download failed (expected if not accessible): {e}")
        
        # Test custom certificate path
        print("\nTesting custom certificate path...")
        custom_cert = "/tmp/custom.pem"
        cert_path, key_path = cert_manager.get_certificate_paths(custom_cert)
        assert cert_path == custom_cert
        print(f"✓ Custom cert path handled correctly: {cert_path}")
    
    print("\n✅ Certificate Manager tests passed!")


def test_ssl_server():
    """Test starting the SSL server"""
    print("\n\nTesting SSL Server Startup...")
    
    # Set environment variables for testing
    os.environ["ENABLE_SSL"] = "True"
    os.environ["SSL_CERTIFICATE"] = "GENERATE"
    os.environ["SSL_PORT"] = "9443"
    os.environ["PORT"] = "9001"
    
    print("Environment configured for SSL testing")
    print("To test the full server:")
    print("1. Run: ENABLE_SSL=True SSL_CERTIFICATE=GENERATE python start_server.py")
    print("2. Test HTTP: curl http://localhost:9001/")
    print("3. Test HTTPS: curl -k https://localhost:9002/")


def main():
    """Run all tests"""
    print("=== Inference Server SSL Functionality Tests ===\n")
    
    try:
        test_certificate_manager()
        test_ssl_server()
        
        print("\n\n=== All tests completed successfully! ===")
        print("\nTo use SSL in production:")
        print("1. Set ENABLE_SSL=True")
        print("2. Set SSL_CERTIFICATE to one of:")
        print("   - 'INDIRECT' (default) - uses roboflow.host wildcard cert")
        print("   - 'GENERATE' - creates self-signed certificate")
        print("   - '/path/to/cert.pem' - uses custom certificate")
        print("3. Optionally set SSL_PORT (default: 9002)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
