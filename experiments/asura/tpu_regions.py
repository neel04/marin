# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where Spot TPUs are cheapest for asura-0 right now, and whether a slice can actually run there.

For v5p and v6e in every region the price list covers, prints the Spot price per chip-hour, the zones that offer the
machine type, the region's preemptible and per-family quota, and whether the Fast Track caches are in that region's
bucket, for the regions where some zone offers the machine: the price list and the quota table also cover regions
where none does. A family that the per-family quota caps, such as v6e, is blocked in every region it does not list.

    uv run python -m experiments.asura.tpu_regions
"""

import json
import subprocess
from dataclasses import dataclass
from typing import Any

import requests

PROJECT = "asura-0"
COMPUTE_SKUS = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"
CACHE = "fasttrack-public-20260926/tokenized/fasttrack-dclm-bpe16k-shuf-20260928/train"
SLICE_CHIPS = 16


@dataclass(frozen=True)
class Generation:
    name: str
    sku_name: str
    machine_type: str
    preemptible_quota: str
    family: str


GENERATIONS = (
    Generation("v5p", "TpuV5p", "ct5p-hightpu-4t", "PREEMPTIBLE-TPU-V5P-per-project-region", "CT5P"),
    Generation("v6e", "TpuV6e", "ct6e-standard-4t", "PREEMPTIBLE-TPU-V6E-per-project-region", "CT6E"),
)


def _gcloud_json(*args: str) -> Any:
    output = subprocess.run(["gcloud", *args, f"--project={PROJECT}", "--format=json"], check=True, capture_output=True)
    return json.loads(output.stdout)


def spot_prices() -> dict[tuple[str, str], float]:
    """Spot price per chip-hour, by generation and region."""
    token = subprocess.run(["gcloud", "auth", "print-access-token"], check=True, capture_output=True, text=True)
    headers = {"Authorization": f"Bearer {token.stdout.strip()}"}
    prices, page = {}, ""
    while True:
        params = {"pageSize": 5000, "pageToken": page, "currencyCode": "USD"}
        data = requests.get(COMPUTE_SKUS, params=params, headers=headers, timeout=60).json()
        for sku in data.get("skus", []):
            for generation in GENERATIONS:
                if sku["description"].startswith(f"{generation.sku_name} attached to Spot Preemptible VMs"):
                    rate = sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"]
                    price = int(rate.get("units", 0)) + rate.get("nanos", 0) / 1e9
                    for region in sku["serviceRegions"]:
                        prices[(generation.name, region)] = price
        page = data.get("nextPageToken", "")
        if not page:
            return prices


def zones_offering(machine_type: str) -> dict[str, list[str]]:
    """The zones, by region, that offer ``machine_type``."""
    zones: dict[str, list[str]] = {}
    for row in _gcloud_json("compute", "machine-types", "list", f"--filter=name={machine_type}"):
        zone = row["zone"].rsplit("/", 1)[-1]
        zones.setdefault(zone.rsplit("-", 1)[0], []).append(zone)
    return zones


def preemptible_limits(quota_id: str) -> dict[str, float]:
    """A preemptible quota's limit by region; a region it lists without a value takes the default."""
    info = _gcloud_json("alpha", "quotas", "info", "describe", quota_id, "--service=compute.googleapis.com")
    values = {
        row.get("dimensions", {}).get("region", "*"): row.get("details", {}).get("value")
        for row in info["dimensionsInfos"]
    }
    default = float(values.get("*") or 0)
    return {region: float(value) if value is not None else default for region, value in values.items()}


def family_limits(family: str) -> dict[str, float] | None:
    """The per-family TPU quota by region, or None if it does not cap ``family``. A capped family gets nothing in a
    region the quota does not list with a value."""
    info = _gcloud_json(
        "alpha",
        "quotas",
        "info",
        "describe",
        "TPUS-PER-TPU-FAMILY-per-project-region",
        "--service=compute.googleapis.com",
    )
    rows = [row for row in info["dimensionsInfos"] if row.get("dimensions", {}).get("tpu_family") == family]
    if not rows:
        return None
    return {
        row["dimensions"]["region"]: float(row["details"]["value"])
        for row in rows
        if "region" in row["dimensions"] and row.get("details", {}).get("value") is not None
    }


def has_cache(region: str) -> bool:
    found = subprocess.run(["gcloud", "storage", "ls", f"gs://marin-{PROJECT}-{region}/{CACHE}/"], capture_output=True)
    return found.returncode == 0


def main() -> None:
    prices = spot_prices()
    rows = []
    for generation in GENERATIONS:
        zones = zones_offering(generation.machine_type)
        preemptible = preemptible_limits(generation.preemptible_quota)
        family = family_limits(generation.family)
        for region, region_zones in zones.items():
            price = prices.get((generation.name, region))
            if price is None:
                continue
            preemptible_limit = preemptible.get(region, preemptible.get("*", 0.0))
            family_limit = None if family is None else family.get(region, 0.0)
            usable = preemptible_limit >= SLICE_CHIPS and (family_limit is None or family_limit >= SLICE_CHIPS)
            rows.append((price, generation.name, region, region_zones, preemptible_limit, family_limit, usable))

    print(f"{'$/chip-h':>9}  {'TPU':4} {'region':16} {'zones':40} {'preempt':>8} {'family':>7}  usable  data")
    for price, name, region, region_zones, preemptible_limit, family_limit, usable in sorted(rows):
        family_text = "-" if family_limit is None else f"{family_limit:.0f}"
        data = "yes" if has_cache(region) else ""
        print(
            f"{price:9.4f}  {name:4} {region:16} {','.join(sorted(region_zones)):40} {preemptible_limit:8.0f}"
            f" {family_text:>7}  {'yes' if usable else '':6}  {data}"
        )


if __name__ == "__main__":
    main()
