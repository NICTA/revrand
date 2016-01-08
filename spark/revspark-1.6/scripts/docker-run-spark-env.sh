#!/bin/sh

env | grep SPARK | awk '{print "export \"" $0 "\""}' > /opt/spark/conf/spark-env.sh

exec $@
