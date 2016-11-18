#!/bin/bash

set -e

echo "Stop and sync NTP"
sudo systemctl stop ntp
sudo ntpd -gq
# Do not start NTP while SDR is running
# It can be started in the future, but ensure there won't be any sudden
#  "jumps" in the timestamps.
sleep 5

echo "Enable bias tee"
/home/pi/rtl-sdr/build/src/rtl_biast -b 1

echo "Starting fastdet"
cd /home/pi/detector
mkdir -p toad/ card/ log/
TOAD_FILE="toad/$(date +"%Y-%m-%d_%H-%M-%S").toad"
CARD_FILE="card/$(date +"%Y-%m-%d_%H-%M-%S").card"
LOG_FILE="log/$(date +"%Y-%m-%d_%H-%M-%S").log"
. ./fastdet.cfg
CARD_ARG=
if [ -n "${EXPORT_CARD}" ]; then
    CARD_ARG="-x ${CARD_FILE}"
fi
exec fastdet \
    -r ${RXID} \
    -t ${THRESH_CARRIER} \
    -u ${THRESH_CORR} \
    -k ${SKIP} \
    -w ${WINDOW} \
    -m ${WISDOM_FILE} \
    -z ${TEMPLATE_FILE} \
    -i rtlsdr \
    -f ${RTL_FREQ} \
    -g ${RTL_GAIN} \
    -o ${TOAD_FILE} ${CARD_ARG} >> ${LOG_FILE}
