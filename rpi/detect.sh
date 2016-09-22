#!/bin/bash

set -e

shutdown() {
    echo "Shutdown"
    kill $(jobs -p)
    exit -1
}

trap "shutdown" SIGINT SIGTERM

echo "Waiting for NTP sync..."
ntp-wait

echo "Starting Thrifty..."
cd /home/pi/detector
rm -f fifo.card buffer
mkfifo fifo.card buffer
TOAD_FILE="toad/$(date +"%Y-%m-%d_%H-%M-%S").toad"

set -m
(
    (thrifty capture buffer >>capture.log || (echo "fastcard died prematurely" && kill 0)) &
    (buffer -m 32M -s 55236 -i buffer -o fifo.card || (echo "buffer died prematurely" && kill 0)) &
    (thrifty detect fifo.card -a $TOAD_FILE >>detect.log || (echo "thrifty detect died prematurely" && kill 0)) &
    wait
)

exit 0
