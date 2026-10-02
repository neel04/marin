#!/usr/bin/env bash
# Startup script for TPU slice hosts: power the host off at the slice's stop time, or once no training worker has
# run for IDLE_MINUTES, so a controller that loses its connection cannot leave the slice billing. The slice's STOP
# termination action and do-nothing failure policy keep a powered-off host stopped.
IDLE_MINUTES=20
stop_at=$(curl -sf --retry 10 -H Metadata-Flavor:Google \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/stop-at)
stop_epoch=$(date -d "$stop_at" +%s)
# Without a readable stop time the host stops at once rather than running unbounded.
if [ -z "$stop_at" ] || [ -z "$stop_epoch" ]; then poweroff; fi
# The bracket keeps the pattern from matching this watchdog's own command line.
systemd-run --unit=slice-watchdog bash -c "
  idle=0
  while sleep 60; do
    if pgrep -f '[e]xperiments.asura.worker' >/dev/null; then idle=0; else idle=\$((idle + 1)); fi
    if [ \$idle -ge $IDLE_MINUTES ] || [ \$(date +%s) -ge $stop_epoch ]; then poweroff; fi
  done"
