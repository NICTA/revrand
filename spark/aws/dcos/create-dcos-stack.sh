#!/bin/bash

set -x
MESOSPHERE_TEMPLATE=https://s3.amazonaws.com/downloads.mesosphere.io/dcos/stable/cloudformation/single-master.cloudformation.json 
TEMPLATE=https://s3-ap-southeast-2.amazonaws.com/dcos-mesos/dcos.single-master.cloudformation.template.json

# Create a Mesosphere DCOS stack on AWS
aws --region ap-southeast-2 cloudformation create-stack --stack-name mesos-spark-revrand --template-url ${TEMPLATE} --parameters ParameterKey=AcceptEULA,ParameterValue="Yes" ParameterKey=KeyName,ParameterValue="dave" ParameterKey=PublicSlaveInstanceCount,ParameterValue="1" ParameterKey=SlaveInstanceCount,ParameterValue="3" --capabilities CAPABILITY_IAM
