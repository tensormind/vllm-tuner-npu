from unittest.mock import MagicMock, patch

from vllm_tuner.utils.cloud_pricing import CloudPricingLookup


class TestCloudPricingLookup:
    def test_no_token_returns_none(self):
        lookup = CloudPricingLookup()
        price = lookup.get_price("gcp", "g2-standard-4", "spot")
        assert price is None

    def test_unknown_provider_returns_none(self):
        lookup = CloudPricingLookup()
        price = lookup.get_price("azure", "some-instance", "spot")
        assert price is None

    def test_no_api_without_token(self):
        with patch("vllm_tuner.settings.settings") as mock_settings:
            mock_settings.VANTAGE_API_TOKEN = None
            lookup = CloudPricingLookup(api_token=None)
            assert not lookup._api_available()

    def test_api_available_with_token(self):
        with patch("vllm_tuner.utils.cloud_pricing._HTTPX_AVAILABLE", True):
            lookup = CloudPricingLookup(api_token="test-token")
            assert lookup._api_available()

    def test_cache_returns_same_value(self):
        lookup = CloudPricingLookup()
        price1 = lookup.get_price("aws", "g6.xlarge", "spot")
        price2 = lookup.get_price("aws", "g6.xlarge", "spot")
        assert price1 == price2

    @patch("vllm_tuner.utils.cloud_pricing._HTTPX_AVAILABLE", True)
    def test_api_fetch_success(self):
        mock_client = MagicMock()
        products_resp = MagicMock()
        products_resp.json.return_value = {"products": [{"id": "aws-ec2-p4d_24xlarge"}]}
        prices_resp = MagicMock()
        prices_resp.json.return_value = {
            "prices": [
                {
                    "region": "us-east-1",
                    "amount": 32.77,
                    "details": {"lifecycle": "on-demand", "platform": "linux"},
                }
            ]
        }
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [products_resp, prices_resp]

        with patch("vllm_tuner.utils.cloud_pricing.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            lookup = CloudPricingLookup(api_token="test-token")
            price = lookup.get_price("aws", "p4d.24xlarge", "on_demand")
            assert price == 32.77

    @patch("vllm_tuner.utils.cloud_pricing._HTTPX_AVAILABLE", True)
    def test_api_empty_products_returns_none(self):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"products": []}
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = resp

        with patch("vllm_tuner.utils.cloud_pricing.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            lookup = CloudPricingLookup(api_token="test-token")
            price = lookup.get_price("aws", "g6.xlarge", "spot")
            assert price is None
