from __future__ import annotations

from vllm_tuner.helper.logging import get_logger


logger = get_logger()

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

_VANTAGE_API_URL = "https://api.vantage.sh/v2"

# Maps our provider names to Vantage service IDs
_SERVICE_MAP = {
    "aws": "aws-ec2",
    "gcp": "gcp-compute-engine",
}

# Default regions used when filtering Vantage API prices
_DEFAULT_REGIONS = {
    "aws": "us-east-1",
    "gcp": "us-central1",
}

# Lifecycle name mapping: our naming → Vantage API naming
_LIFECYCLE_MAP = {
    "spot": "spot",
    "on_demand": "on-demand",
}


class CloudPricingLookup:
    """Looks up cloud instance pricing via the Vantage API.

    Queries https://api.vantage.sh/v2 for live pricing data.
    Requires ``VANTAGE_API_TOKEN`` env var and httpx installed
    (``pip install 'llm-vllm-tuner[http]'``).
    Returns ``None`` when the API is unavailable — cost analysis is skipped.

    Note:
        The Vantage Products API currently provides catalog prices for AWS only.
        GCP "support" on the Vantage pricing page refers to *cost tracking*
        (connecting billing via BigQuery), not the instance catalog API.
        For GCP instances, set ``price_per_hour`` in ``CostConfig`` directly.

    """

    def __init__(self, *, api_token: str | None = None, region: str | None = None):
        if api_token is None:
            from vllm_tuner.settings import settings

            token = settings.VANTAGE_API_TOKEN
            self._token = token.get_secret_value() if token else None
        else:
            self._token = api_token
        self._region = region
        self._cache: dict[str, float | None] = {}

    def _api_available(self) -> bool:
        return _HTTPX_AVAILABLE and self._token is not None

    def _fetch_price_from_api(self, cloud_provider: str, instance_type: str, pricing_mode: str) -> float | None:
        """Query Vantage API for a single instance price."""
        service_id = _SERVICE_MAP.get(cloud_provider)
        if service_id is None:
            return None

        region = self._region or _DEFAULT_REGIONS.get(cloud_provider, "")
        lifecycle = _LIFECYCLE_MAP.get(pricing_mode, pricing_mode)

        try:
            with httpx.Client(timeout=10) as client:
                # Step 1: resolve product ID
                resp = client.get(
                    f"{_VANTAGE_API_URL}/products",
                    params={
                        "provider_id": cloud_provider,
                        "service_id": service_id,
                        "name": instance_type,
                    },
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                resp.raise_for_status()
                products = resp.json().get("products", [])
                if not products:
                    return None

                product_id = products[0]["id"]

                # Step 2: get prices for this product
                resp = client.get(
                    f"{_VANTAGE_API_URL}/products/{product_id}/prices",
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                resp.raise_for_status()
                prices = resp.json().get("prices", [])

            # Find matching price by region + lifecycle + linux platform
            for price_entry in prices:
                details = price_entry.get("details", {})
                entry_region = price_entry.get("region", "")
                entry_lifecycle = details.get("lifecycle", "")
                entry_platform = details.get("platform", "")
                amount = float(price_entry.get("amount", 0))

                if (
                    entry_region == region
                    and entry_lifecycle == lifecycle
                    and entry_platform in ("linux", "linux-enterprise", "")
                    and amount > 0
                ):
                    return amount

            # If linux price is zero or missing, try other platforms in same region
            for price_entry in prices:
                details = price_entry.get("details", {})
                amount = float(price_entry.get("amount", 0))
                if price_entry.get("region") == region and details.get("lifecycle") == lifecycle and amount > 0:
                    return amount

            # Last resort: any region, matching lifecycle, non-zero price
            for price_entry in prices:
                details = price_entry.get("details", {})
                amount = float(price_entry.get("amount", 0))
                if details.get("lifecycle") == lifecycle and amount > 0:
                    return amount

        except Exception:
            logger.debug(
                "Vantage API request failed for {}/{}",
                cloud_provider,
                instance_type,
            )
        return None

    def get_price(self, cloud_provider: str, instance_type: str, pricing_mode: str = "spot") -> float | None:
        """Get hourly price for an instance type."""
        cache_key = f"{cloud_provider}/{instance_type}/{pricing_mode}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        price: float | None = None

        if self._api_available():
            price = self._fetch_price_from_api(cloud_provider, instance_type, pricing_mode)

        if price is None:
            logger.info(
                "CloudPricingLookup: no pricing available for {}/{}/{}, cost analysis will be skipped",
                cloud_provider,
                instance_type,
                pricing_mode,
            )

        self._cache[cache_key] = price
        return price
