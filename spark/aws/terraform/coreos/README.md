Running Mesos inside docker
===========================

Use Mesospheres master and slave docker images. Use mesos docker containerizer to then run spark (docker insider docker). Need to mount a some dirs/libs when running the outer docker for the inner one to work.

Provision with terraform.

Currently fails when Mesos tries to run up the revrand-spark image and execute spark.

```bash
docker -H unix:///var/run/docker.sock run -c 2048 -m 1476395008 -e SPARK_EXECUTOR_OPTS= -e SPARK_USER=root -e SPARK_EXECUTOR_MEMORY=1024m -e MESOS_SANDBOX=/mnt/mesos/sandbox -e MESOS_CONTAINER_NAME=mesos-6c355814-d122-4e37-9d9c-f055a5da3bb9-S0.6ca7d7a5-e47e-43bf-9422-7b1e79c02c64 -v /tmp/mesos/slaves/6c355814-d122-4e37-9d9c-f055a5da3bb9-S0/frameworks/6c355814-d122-4e37-9d9c-f055a5da3bb9-0275/executors/0/runs/6ca7d7a5-e47e-43bf-9422-7b1e79c02c64:/mnt/mesos/sandbox --net host --entrypoint /bin/sh --name mesos-6c355814-d122-4e37-9d9c-f055a5da3bb9-S0.6ca7d7a5-e47e-43bf-9422-7b1e79c02c64 dtpc/revrand-spark -c  "/opt/spark/bin/spark-class" org.apache.spark.executor.CoarseGrainedExecutorBackend --driver-url spark://CoarseGrainedScheduler@10.0.0.54:7001 --executor-id 6c355814-d122-4e37-9d9c-f055a5da3bb9-S0 --hostname 10.0.0.91 --cores 2 --app-id 6c355814-d122-4e37-9d9c-f055a5da3bb9-0275

Exception in thread "main" java.lang.IllegalArgumentException: Not enough arguments: missing class name.
at org.apache.spark.launcher.CommandBuilderUtils.checkArgument(CommandBuilderUtils.java:242)
at org.apache.spark.launcher.Main.main(Main.java:51)
```
