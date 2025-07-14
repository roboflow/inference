"""SSL Certificate Manager for Inference Server

This module handles SSL certificate management including:
- Downloading roboflow.host wildcard certificates
- Generating self-signed certificates
- Certificate caching and rotation
"""

import os
import ssl
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import requests
import tempfile
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from inference.core import logger
from inference.core.env import MODEL_CACHE_DIR


class SSLCertificateManager:
    """Manages SSL certificates for the Inference Server"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or MODEL_CACHE_DIR) / "ssl_certificates"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._cert_lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 86400  # 24 hours in seconds
        
    def get_certificate_paths(self, certificate_mode: str) -> Tuple[str, str]:
        """Get paths to certificate and key files based on mode
        
        Args:
            certificate_mode: One of "INDIRECT", "GENERATE", or a custom path
            
        Returns:
            Tuple of (cert_path, key_path)
        """
        if certificate_mode == "INDIRECT":
            return self._get_roboflow_host_certificate()
        elif certificate_mode == "GENERATE":
            return self._get_self_signed_certificate()
        else:
            # Custom certificate path provided
            cert_path = certificate_mode
            # Assume key is in same directory with .key extension
            key_path = cert_path.replace('.pem', '.key').replace('.crt', '.key')
            if not os.path.exists(key_path):
                # Try common naming patterns
                key_path = cert_path.replace('fullchain', 'privkey').replace('cert', 'key')
            return cert_path, key_path

    def _get_roboflow_host_certificate(self) -> Tuple[str, str]:
        """Download or retrieve cached roboflow.host wildcard certificate"""
        cert_path = self.cache_dir / "roboflow_host_fullchain.pem"
        key_path = self.cache_dir / "roboflow_host_privkey.pem"
        
        # Check if we need to refresh the certificate
        current_time = time.time()
        should_refresh = (
            not cert_path.exists() or 
            not key_path.exists() or
            (current_time - self._last_check_time) > self._check_interval
        )
        
        if should_refresh:
            with self._cert_lock:
                # Double-check after acquiring lock
                if (current_time - self._last_check_time) > self._check_interval:
                    self._download_roboflow_host_certificate(cert_path, key_path)
                    self._last_check_time = current_time
                    
        return str(cert_path), str(key_path)
    
    def _download_roboflow_host_certificate(self, cert_path: Path, key_path: Path):
        """Download certificate files from roboflow.host"""
        logger.info("Downloading roboflow.host wildcard certificate...")
        
        try:
            # Download certificate
            cert_response = requests.get(
                "https://roboflow.host/certificate/fullchain.pem",
                timeout=30
            )
            cert_response.raise_for_status()
            
            # Download private key
            key_response = requests.get(
                "https://roboflow.host/certificate/privkey.pem",
                timeout=30
            )
            key_response.raise_for_status()
            
            # Write to temporary files first, then move atomically
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as tmp_cert:
                tmp_cert.write(cert_response.text)
                tmp_cert_path = tmp_cert.name
                
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as tmp_key:
                tmp_key.write(key_response.text)
                tmp_key_path = tmp_key.name
            
            # Move files atomically
            os.replace(tmp_cert_path, cert_path)
            os.replace(tmp_key_path, key_path)
            
            # Set appropriate permissions
            os.chmod(key_path, 0o600)
            
            logger.info("Successfully downloaded roboflow.host certificate")
            
        except Exception as e:
            logger.error(f"Failed to download roboflow.host certificate: {e}")
            # If download fails and we have existing certs, keep using them
            if cert_path.exists() and key_path.exists():
                logger.info("Using existing cached certificates")
            else:
                # Fall back to self-signed certificate for development
                logger.warning("No cached certificate available, falling back to self-signed certificate")
                self._generate_self_signed_certificate(cert_path, key_path)

    def _get_self_signed_certificate(self) -> Tuple[str, str]:
        """Generate or retrieve cached self-signed certificate"""
        cert_path = self.cache_dir / "self_signed.pem"
        key_path = self.cache_dir / "self_signed.key"
        
        if not cert_path.exists() or not key_path.exists():
            with self._cert_lock:
                # Double-check after acquiring lock
                if not cert_path.exists() or not key_path.exists():
                    self._generate_self_signed_certificate(cert_path, key_path)
                    
        return str(cert_path), str(key_path)
    
    def _generate_self_signed_certificate(self, cert_path: Path, key_path: Path):
        """Generate a self-signed certificate"""
        logger.info("Generating self-signed certificate...")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Roboflow Inference"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("127.0.0.1"),
                    x509.DNSName("*.roboflow.host"),
                ]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )
        
        # Write private key
        with open(key_path, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                )
            )
        
        # Write certificate
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        # Set appropriate permissions
        os.chmod(key_path, 0o600)
        
        logger.info("Successfully generated self-signed certificate")
