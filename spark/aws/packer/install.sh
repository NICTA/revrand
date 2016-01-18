
sleep 30

## Mesos
sudo echo "deb http://repos.mesosphere.io/ubuntu/ trusty main" | \
    sudo tee /etc/apt/sources.list.d/mesosphere.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF
sudo apt-get -y update
sudo apt-get -y install mesos

# disable services
sudo echo manual | sudo tee /etc/init/mesos-master.override
sudo echo manual | sudo tee /etc/init/mesos-slave.override
sudo echo manual | sudo tee /etc/init/zookeeper.override

## Docker
sudo apt-get -y install docker.io
sudo usermod -aG docker ubuntu

## locale
sudo locale-gen en_AU.UTF-8

## Speed up SSH (remove message-of-the-day) 
sudo rm /etc/update-motd.d/{10-help-text,50-landscape-sysinfo,51-cloudguest,90-updates-available,91-release-upgrade}

