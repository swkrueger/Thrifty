#!/bin/sh

# Wait for internet connectivity and reload after connectivity has been established
# Based on http://unix.stackexchange.com/a/249609

host="${1:-8.8.8.8}"

pingcheck() {
    ping -n -c 1 -w 5 $1 >/dev/null 2>&1
}

while :; do
    pingcheck ${host} && break
    sleep 10
done

sudo systemctl restart ntp
