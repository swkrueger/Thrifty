# Raspberry Pi 3 Installation Guide

## Prepare SD-card
On Linux PC:
```
#!bash

wget http://director.downloads.raspberrypi.org/raspbian_lite/images/raspbian_lite-2016-05-31/2016-05-27-raspbian-jessie-lite.zip
sha1sum 2016-05-27-raspbian-jessie-lite.zip
unzip 2016-05-27-raspbian-jessie-lite.zip
sudo dd bs=4M if=2016-05-27-raspbian-jessie-lite.img of=/dev/mmcblk0
sync
```

References:

 - [Raspbian download page](https://www.raspberrypi.org/downloads/raspbian/)
 - [Installing images: Linux](https://www.raspberrypi.org/documentation/installation/installing-images/linux.md)

## Rpi: First boot
### raspi-config
Run `sudo raspi-config` and set:

 - `1 Expand Filesystem`
 - `2 Change user password`
 - `9 Advanced Options -> A2 Hostname`: set it to `rpi-rxX` where X is the receiver ID
 - `5 Internationalisation Options -> I1 Change Locale`: Add `en_US.UTF-8` and `en_ZA.UTF-8`
 - `5 Internationalisation Options -> I2 Change Timezone`: `Africa -> Johannesburg`

### Setup WiFi connectivity (Optional)
To connect to WiFi:

Edit `/etc/wpa_supplicant/wpa_supplicant.conf`:
```
network={
    ssid="mywifi"
    psk="mypassword"
}
```

See also: [Setting wifi up via the command line](https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md).

### Update software

```
#!bash

sudo apt update && sudo apt upgrade
```


## Install packages
Install tools:
```
sudo apt install vim rsync screen htop git buffer
```

Install fastcard dependencies:
```
sudo apt install build-essential cmake libfftw3-dev rtl-sdr librtlsdr-dev
```

Install libvolk from jessie-backports:
```
echo "deb http://debian.mirror.ac.za/debian jessie-backports main" | sudo tee /etc/apt/sources.list.d/jessie-backports.list
sudo apt update
sudo apt install libvolk1.3 libvolk1-dev
sudo rm /etc/apt/sources.list.d/jessie-backports.list
sudo apt update
```
(note: it [isn't recommended](https://www.raspbian.org/RaspbianFAQ#Can_I_mix_packages_from_the_Debian_repositories_with_Raspbian.3F) to mix packages from the Raspbian repository and the official Debian repositories, but this seems to work ok)

Install thrifty dependencies:
```
sudo apt install python-pip python-numpy python-scipy python-matplotlib
```


## Install thrifty
### Setup repository access

 - Generate SSH key: run `ssh-keygen`
 - Copy public key: `cat ~/.ssh/id_rsa.pub`
 - Add key to deployment keys

### Clone repository
Cd to home directory and run:
```
git clone https://github.com/swkrueger/Thrifty.git thrifty
```

### Install pyfftw3
FFTW is built without double precision support (fftwl) on the Raspberry Pi. It is necessary to patch pyfftw to remove support for fftwl. (Inspired by [this](https://web.archive.org/web/20160502002638/http://www.eliteraspberries.com/blog/2013/10/building-a-numerical-python-environment-in-debian.html))

Get pyFFTW source c ode:
```
mkdir ~/build
cd ~/build
wget https://pypi.python.org/packages/source/p/pyFFTW/pyFFTW-0.9.2.tar.gz
tar xzvf pyFFTW-0.9.2.tar.gz
cd pyFFTW-0.9.2
```

Patch:
```
patch -p1 < ~/thrifty/rpi/pyFFTW-0.9.2-no-fftwl.patch
```

Install Cython compiler and compile Cython code:
```
sudo apt install cython
cython pyfftw/pyfftw.pyx
```

Build and install pyFFTW:
```
python setup.py build
sudo python setup.py install
```

Verify that pyfftw is installed:
```
cd && python -c 'import pyfftw; print pyfftw.version'
```

### Compile fastcard
```
mkdir -p ~/build/fastcard && cd ~/build/fastcard
cmake ~/thrifty/fastcard -DTUNE_BUILD=0 -DCMAKE_C_FLAGS="-mcpu=cortex-a53 -mfpu=neon-vfpv4"
make && sudo make install
```

### Compile fastdet
```
mkdir -p ~/build/fastdet && cd ~/build/fastdet
cmake ~/thrifty/fastdet -DCMAKE_C_FLAGS="-mcpu=cortex-a53 -mfpu=neon-vfpv4"
make && sudo make install
```

### Install thrifty
```
cd ~/thrifty
sudo pip install --no-deps -e .
```

## Setup remote access
### Weaved
Based on instructions [here](https://www.weaved.com/installing-weaved-raspberry-pi-raspbian-os/):
```
sudo apt-get install weavedconnectd
sudo weavedinstaller
```

### Reverse SSH
An alternative to weaved is to install a reverse SSH connection to a remote host with a public IP. For this setup we assume the ssh daemon on the remote host has been configured to accept port forwarding for the `tunnel@public.example.org`.

On the remote host, add the Pi3's public key to `~tunnel/.ssh/authorized_keys`.

To test, establish the connection:
```
ssh -CNnT -R 0.0.0.0:5001:localhost:22 tunnel@public.example.org
```
and from another PC, try SSHing to the Pi:
```
ssh public.example.org -p 5001
```

To persist the connection by keeping it alive and automatically starting it after a reboot, install autossh:
```
sudo apt install autossh
```
and add a systemd service by creating the file `/etc/systemd/system/ssh-tunnel.service`:
```
[Unit]
Description=Establish reverse tunnel to local SSH port
After=network.target

[Service]
Environment="AUTOSSH_GATETIME=0"
ExecStart=/usr/bin/autossh -M 0 -o "ServerAliveInterval 60" -o "ServerAliveCountMax 3" -o "ExitOnForwardFailure yes" -CNnT -R 0.0.0.0:5001:localhost:22 tunnel@public.example.org
User=pi
Group=pi

[Install]
WantedBy=multi-user.target
```
Then, reload, start the service, and enable during boot time:
```
sudo systemctl daemon-reload
sudo systemctl start ssh-tunnel.service
sudo systemctl enable ssh-tunnel.service
```


## Setup 3G internet

## Speed up
### Disable unnecessary hardware
Some of the peripherals that aren't being used can be used to conserve power.

To disable HDMI on boot, add the following to `/etc/rc.local`:
```
# Disable HDMI
/usr/bin/tvservice -o
```

Disable WiFi and Bluetooth by blacklisting it. Create the file `/etc/modprobe.d/wifi-bluetooth-blacklist.conf` with
```
# Disable WiFi
blacklist brcmfmac
blacklist brcmutil

# Disable Bluetooth
blacklist btbcm
blacklist hci_uart
```
(I'm not sure whether this will actually conserve any power since it isn't actually powering down the devices but only disabling the kernel modules).

### Boost maximum USB power
By default, the maximum total current that all USB peripherals may draw is 600mA ([source](https://raspberrypi.stackexchange.com/questions/27708/is-setting-max-usb-current-1-to-give-more-power-to-usb-devices-a-bad-idea)). This can be increased to 1.2A by adding the following to `/boot/config.txt`:
```
# Boost maximum total USB current to 1.2A
max_usb_current=1
```


## Services
### Detector
```
mkdir -p /home/pi/detector/{toad,card,log}
cd
cp thrifty/rpi/{detector.cfg,template.npy,fastdet.cfg,template.tpl} detector/
sudo cp thrifty/rpi/detector.service /etc/systemd/system/
sudo chown root:root /etc/systemd/system/detector.service
sudo chmod 644 /etc/systemd/system/detector.service
```

Change `rxid` in `detector/detector.cfg` and `detector/fastdet.cfg`.

Start and enable the service:
```
sudo systemctl daemon-reload
sudo systemctl start detector.service
sudo systemctl enable detector.service
```

### Uploader
Very simple detection uploader: rsync detections every 10 minutes to server.
Run `crontab -e` and add:
```
*/10 * * * * /usr/bin/rsync -e ssh -avzq --log-file=/home/pi/rsync.log /home/pi/detector/toad/ uploader@myserver:rpi-rxX
```
where X is the receiver ID.

### Force NTP restart
The 3G modem creates a USB network interface, which causes the NTP service to start before a 3G connection has been established.
Here is a hack to check for internet connectivity and reload NTP after a connection has been established:
```
cd ~/thrifty/rpi
sudo install -m 755 ntp-after-online.sh /usr/local/bin/
sudo install -m 644 ntp-after-online.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ntp-after-online.service
```
