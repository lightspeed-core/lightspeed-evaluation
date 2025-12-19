"""Unit tests for SSL certificate bundle handling."""

from pathlib import Path

from lightspeed_evaluation.core.system.ssl_certifi import (
    create_ssl_certifi_bundle,
    get_ssl_cert_files_paths_from_system_yaml,
    get_system_ssl_cert_file,
    _get_unique_ssl_cert_paths,
)


class TestGetSslCertFilesPathsFromSystemYaml:
    """Tests for extracting SSL cert paths from config data."""

    def test_extracts_cert_when_ssl_verify_true(self):
        """Test extracting SSL cert when ssl_verify is True."""
        config = {
            "ssl_verify": True,
            "ssl_cert_file": "/path/to/cert.pem",
        }

        result = get_ssl_cert_files_paths_from_system_yaml(config)

        assert result == ["/path/to/cert.pem"]

    def test_ignores_cert_when_ssl_verify_false(self):
        """Test that ssl_cert_file is ignored when ssl_verify is False."""
        config = {
            "ssl_verify": False,
            "ssl_cert_file": "/path/to/cert.pem",
        }

        result = get_ssl_cert_files_paths_from_system_yaml(config)

        assert result == []

    def test_nested_configs(self):
        """Test extracting SSL certs from nested configuration."""
        config = {
            "service_a": {
                "ssl_verify": True,
                "ssl_cert_file": "/path/to/cert_a.pem",
            },
            "service_b": {
                "ssl_verify": True,
                "ssl_cert_file": "/path/to/cert_a.pem",
            },
            "service_c": {
                "ssl_verify": False,
                "ssl_cert_file": "/path/to/cert_b.pem",
            },
            "service_d": {
                "ssl_verify": True,
                "ssl_cert_file": "/path/to/cert_c.pem",
            },
        }

        result = get_ssl_cert_files_paths_from_system_yaml(config)
        assert result == [
            "/path/to/cert_a.pem",
            "/path/to/cert_a.pem",
            "/path/to/cert_c.pem",
        ]


class TestGetSystemSslCertFile:
    """Tests for getting system SSL cert file from environment."""

    def test_returns_cert_file_when_env_set(self, mocker):
        """Test when SSL_CERT_FILE environment variable is set."""
        mocker.patch.dict("os.environ", {"SSL_CERT_FILE": "/system/cert.pem"})

        result = get_system_ssl_cert_file()

        assert result == ["/system/cert.pem"]

    def test_returns_empty_when_env_not_set(self, mocker):
        """Test when SSL_CERT_FILE environment variable is not set."""
        mocker.patch.dict("os.environ", {}, clear=True)

        result = get_system_ssl_cert_file()

        assert result == []


class TestGetUniqueSslCertPaths:
    """Tests for getting unique SSL certificate paths."""

    def test_returns_unique_paths(self):
        """Test that duplicate paths are removed."""
        cert_paths = [
            "/path/to/cert_a.pem",
            "/path/to/cert_b.pem",
            "/path/to/cert_a.pem",
        ]

        result = _get_unique_ssl_cert_paths(cert_paths)

        assert set(result) == {"/path/to/cert_a.pem", "/path/to/cert_b.pem"}

    def test_returns_empty_list_when_no_paths(self):
        """Test that an empty list is returned when no paths are provided."""
        result = _get_unique_ssl_cert_paths([])

        assert result == []


class TestCreateSslCertifiBundle:
    """Tests for creating combined SSL certificate bundle."""

    def test_returns_certifi_bundle_when_no_custom_certs(self, mocker):
        """Test that certifi bundle is returned when no custom certs exist."""
        mocker.patch.dict("os.environ", {}, clear=True)

        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = "/path/to/certifi/cacert.pem"

        result = create_ssl_certifi_bundle({})

        assert result == "/path/to/certifi/cacert.pem"

    def test_combines_certifi_with_custom_cert(self, mocker, tmp_path):
        """Test combining certifi bundle with custom cert from config."""
        mocker.patch.dict("os.environ", {}, clear=True)

        certifi_bundle = tmp_path / "certifi.pem"
        certifi_bundle.write_text("CERTIFI BUNDLE\n")

        custom_cert = tmp_path / "custom.pem"
        custom_cert.write_text("CUSTOM CERT\n")

        config = {
            "ssl_verify": True,
            "ssl_cert_file": str(custom_cert),
        }

        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = str(certifi_bundle)

        result = create_ssl_certifi_bundle(config)
        content = Path(result).read_text()

        assert "CERTIFI BUNDLE" in content
        assert "CUSTOM CERT" in content

    def test_combines_config_and_env_certs(self, mocker, tmp_path):
        """Test combining certs from both config and environment."""
        certifi_bundle = tmp_path / "certifi.pem"
        certifi_bundle.write_text("CERTIFI BUNDLE\n")

        config_cert = tmp_path / "config.pem"
        config_cert.write_text("CONFIG CERT\n")

        env_cert = tmp_path / "env.pem"
        env_cert.write_text("ENV CERT\n")

        mocker.patch.dict("os.environ", {"SSL_CERT_FILE": str(env_cert)})

        config = {
            "ssl_verify": True,
            "ssl_cert_file": str(config_cert),
        }

        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = str(certifi_bundle)

        result = create_ssl_certifi_bundle(config)
        content = Path(result).read_text()

        assert "CERTIFI BUNDLE" in content
        assert "CONFIG CERT" in content
        assert "ENV CERT" in content

    def test_registers_atexit_cleanup(self, mocker, tmp_path):
        """Test that atexit cleanup is registered for temp bundle."""
        mocker.patch.dict("os.environ", {})

        certifi_bundle = tmp_path / "certifi.pem"
        certifi_bundle.write_text("CERTIFI BUNDLE\n")

        custom_cert = tmp_path / "custom.pem"
        custom_cert.write_text("CUSTOM CERT\n")

        config = {"ssl_verify": True, "ssl_cert_file": str(custom_cert)}

        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_atexit = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.atexit.register"
        )
        mock_where.return_value = str(certifi_bundle)

        create_ssl_certifi_bundle(config)

        assert mock_atexit.called
