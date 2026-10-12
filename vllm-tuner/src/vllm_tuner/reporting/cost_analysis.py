from __future__ import annotations

import math

from vllm_tuner.core.models import BenchmarkResult, CostConfig, CostReport
from vllm_tuner.helper.logging import get_logger
from vllm_tuner.utils.cloud_pricing import CloudPricingLookup


logger = get_logger()


class CostAnalyzer:
    """Analyzes cost-performance trade-offs for tuning results."""

    def __init__(self):
        self._pricing = CloudPricingLookup()

    def analyze(self, benchmark: BenchmarkResult, cost_config: CostConfig) -> CostReport:
        """Compute cost analysis for a benchmark result."""
        if not cost_config.target_throughput or not cost_config.instance_type:
            logger.info("CostAnalyzer: insufficient config for cost analysis (no target_throughput or instance_type)")
            return CostReport()

        # Use explicit price from config if provided, otherwise query Vantage API
        price = cost_config.price_per_hour
        if price is None:
            price = self._pricing.get_price(
                cost_config.cloud_provider,
                cost_config.instance_type,
                cost_config.pricing_mode,
            )
        if price is None:
            logger.info(
                "CostAnalyzer: no pricing data for {}/{}, skipping cost analysis",
                cost_config.cloud_provider,
                cost_config.instance_type,
            )
            return CostReport(cloud_provider=cost_config.cloud_provider, instance_type=cost_config.instance_type)

        throughput = benchmark.throughput_req_per_sec
        if throughput <= 0:
            return CostReport(cloud_provider=cost_config.cloud_provider, instance_type=cost_config.instance_type)

        instances_needed = math.ceil(cost_config.target_throughput / throughput)
        hourly_cost = instances_needed * price
        monthly_cost = hourly_cost * 730  # ~30.4 days
        perf_per_dollar = throughput / price if price > 0 else 0.0

        report = CostReport(
            instances_needed=instances_needed,
            hourly_cost=hourly_cost,
            monthly_cost=monthly_cost,
            perf_per_dollar=perf_per_dollar,
            cloud_provider=cost_config.cloud_provider,
            instance_type=cost_config.instance_type,
            pricing_mode=cost_config.pricing_mode,
        )

        logger.info(
            "CostAnalyzer: {} instances × ${:.2f}/hr = ${:.2f}/hr (${:.0f}/mo), perf/$ = {:.2f}",
            report.instances_needed,
            price,
            report.hourly_cost,
            report.monthly_cost,
            report.perf_per_dollar,
        )
        return report
