# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute a serialized Marin step on a GCP VM."""

import sys

import cloudpickle
import fsspec


def main() -> None:
    with fsspec.open(sys.argv[1], "rb") as payload:
        entrypoint = cloudpickle.load(payload)
    entrypoint.callable(*entrypoint.args, **entrypoint.kwargs)


if __name__ == "__main__":
    main()
