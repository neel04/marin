# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tarfile
from pathlib import Path

import pytest

from experiments.asura import tpu_queue
from experiments.asura.tpu_queue import Run, run_queue

EUROPE_FLEX = "experiments/asura/fast_track_dense_ladder_v6e_eu_flex.yaml"
US_SOUTH = "experiments/asura/fast_track_dense_ladder_south.yaml"


@pytest.fixture
def fake_slices(monkeypatch, tmp_path):
    """Slices that come up in the zones listed in ``capacity`` and nowhere else, and runs that always finish."""
    events: list[tuple[str, str]] = []
    capacity: set[str] = set()
    created: set[str] = set()

    def create(settings, stop_at):
        events.append(("create", settings.zone))
        if settings.zone not in capacity:
            raise TimeoutError(f"{settings.tpu_vm} not ready")
        created.add(settings.zone)

    def delete(settings):
        events.append(("delete", settings.zone))
        created.discard(settings.zone)

    def train(run, launch, snapshot, timeout):
        config = next(arg for arg in launch if arg.startswith("--config="))
        events.append(("train", f"{run.run_id} {Path(config.split('=', 1)[1]).name}"))
        return True

    monkeypatch.setattr(tpu_queue, "create", create)
    monkeypatch.setattr(tpu_queue, "delete", delete)
    monkeypatch.setattr(tpu_queue, "setup", lambda settings, archive: None)
    monkeypatch.setattr(tpu_queue, "_hosts_ready", lambda settings, count: settings.zone in created)
    monkeypatch.setattr(tpu_queue, "archive_checkout", lambda archive: _empty_archive(archive))
    monkeypatch.setattr(tpu_queue, "_train", train)
    monkeypatch.setattr(tpu_queue, "PROVISION_RETRY_WAIT", tpu_queue.timedelta(0))
    return events, capacity


def _empty_archive(path: Path) -> None:
    with tarfile.open(path, "w:gz"):
        pass


def test_queue_moves_to_the_next_region_when_the_cheapest_gets_no_hosts(fake_slices):
    events, capacity = fake_slices
    capacity.add("europe-west4-a")
    run = Run("fasttrack-x-d768-dclmbpe-{accelerator}-20261006")

    finished = run_queue([Path(US_SOUTH), Path(EUROPE_FLEX)], "d768", [run], run_minutes=9)

    assert finished == [run]
    trains = [detail for kind, detail in events if kind == "train"]
    # The run trained once, in the second region, under that region's accelerator and config.
    assert trains == ["fasttrack-x-d768-dclmbpe-v6e16-20261006 queue-config-1.yaml"]
    # The cheapest region was tried first, and its slice deleted before the queue moved on.
    tried_south, tried_europe = events.index(("create", "us-south1-a")), events.index(("create", "europe-west4-a"))
    assert tried_south < tried_europe
    assert ("delete", "us-south1-a") in events[tried_south:tried_europe]
    # Every region's slice is deleted at the end.
    assert {"us-south1-a", "europe-west4-a"} <= {zone for kind, zone in events[-2:] if kind == "delete"}


def test_queue_stays_in_the_cheapest_region_when_it_has_hosts(fake_slices):
    events, capacity = fake_slices
    capacity.update({"us-south1-a", "europe-west4-a"})
    runs = [
        Run("fasttrack-a-d768-dclmbpe-{accelerator}-20261006"),
        Run("fasttrack-b-d768-dclmbpe-{accelerator}-20261006"),
    ]

    assert run_queue([Path(US_SOUTH), Path(EUROPE_FLEX)], "d768", runs, run_minutes=9) == runs
    assert [kind for kind, zone in events if zone == "europe-west4-a" and kind == "create"] == []
    trains = [detail for kind, detail in events if kind == "train"]
    assert trains == [
        "fasttrack-a-d768-dclmbpe-v5p32-20261006 queue-config-0.yaml",
        "fasttrack-b-d768-dclmbpe-v5p32-20261006 queue-config-0.yaml",
    ]
