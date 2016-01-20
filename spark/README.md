Scripts for deploying a Spark cluster for revrand
=================================================

NOTE: Work in progress 

Experimenting with running a Mesos cluster and using Mesos' Spark and docker containeriser support.

Docker images: 
Mesos spark docker image `dtpc/revrand-spark`
Spark driver docker image `dtpc/revrand-spark-jupyter`


Mesosphere DCOS
---------------

`cd aws/dcos`, then run `create-dcos-stack.sh`

Automatically sets up a Mesos cluster with DNS, Marathon and zookeeper.

Need to ssh into one of the machines and manually run revrand-spark-jupyter docker container to get jupyter spark driver. Should be able to submit this to marathon directly.


Hashicorp Terraform
-------------------

`cd aws/terraform/custom`, then create `terraform.tfvars` file with following filled in with your AWS credentials:
```
aws_access_key = ""
aws_secret_key = ""
aws_key_path = "~/.ssh/<key_file>.pem"
aws_key_name = "<key_name>"
```
Run `terraform apply`

This takes a few minutes, after which the Mesos master and Jupiter notebook web addresses are output to the terminal.

In Jupiter start a new PySpark kernel and the spark context is available as `sc`. When a spark task is sent to the Mesos master, it runs the `revrand-spark` docker image on each slave. The first time this is done it needs to download thee image first, so it takes a few mins for the task to get started (tasks will appear as STAGING on Mesos while this happens).

Issues
------
* revrand sgd_spark pickling error
* Networking of spark cluster
* Runing multiple notebooks

