Scripts for deploying a Spark cluster for revrand
=================================================

NOTE: Work in progress 

Experimenting with running a Mesos cluster and using Mesos' Spark and docker containeriser support.

Docker images: 
Mesos spark docker image `dtpc/revrand-spark`
Spark driver docker image `dtpc/revrand-spark-jupyter`


Deployment methods
------------------

### Mesosphere DCOS

https://mesosphere.com/product/

`cd aws/dcos`, then run `create-dcos-stack.sh`.

Automatically sets up a Mesos cluster with DNS, Marathon and zookeeper. This uses AWS cloudformation with the Mesosphere DCOS template.

Once deployed (you can check via the AWS web console or by running `aws cloudformation --describe-stacks`), submit the jupyter docker container as a task to Marathon by running:

`./run-pyspark-jupyter.sh <mesos_master_private_ip>:5050`

where the Mesos master private ip will be something like 10.0.x.x. This can can be obtained from the AWS console. 

Jupyter will take a minute or two to startup, and can then be accessed from on public slave IP (from AWS ECS console) and port 9999. Mesos can be accessed on the master public IP on port 5050. 

### Hashicorp Terraform

https://www.terraform.io/

Install Terraform.

`cd aws/terraform/custom`, then create `terraform.tfvars` file with following filled in with your AWS credentials:
```
aws_access_key = ""
aws_secret_key = ""
aws_key_path = "~/.ssh/<key_file>.pem"
aws_key_name = "<key_name>"
```
Run `terraform apply`

This takes a few minutes, after which the Mesos master and Jupiter notebook web addresses are output to the terminal.

Jupyter Notebook
----------------
In Jupiter start the `example-revrand-sgd.ipynb` which runs revrands AdaDelta SGD algorithm in parallel to optimise weights of a radial basis function to fit data from a sine wave.

Alternatively start a new PySpark kernel and the spark context is available as `sc`. When a spark task is sent to the Mesos master, it runs the `revrand-spark` docker image on each slave. The first time this is done it needs to download thee image first, so it takes a few mins for the task to get started (tasks will appear as STAGING on Mesos while this happens).


Issues
------
* revrand sgd_spark pickling error
* Networking of spark cluster (security)
* Runing multiple notebooks
