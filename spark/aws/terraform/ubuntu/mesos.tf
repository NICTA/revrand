/*
Master
*/



variable "slave_ips" {
  default = {
    "0" = "10.0.0.100"
    "1" = "10.0.0.101"
    "2" = "10.0.0.102"
    "3" = "10.0.0.103"
    "4" = "10.0.0.104"
    "5" = "10.0.0.105"
    "6" = "10.0.0.106"
    "7" = "10.0.0.107"
    "8" = "10.0.0.108"
    "9" = "10.0.0.109"
    "10" = "10.0.0.110"
    "11" = "10.0.0.111"
    "12" = "10.0.0.112"
    "13" = "10.0.0.113"
    "14" = "10.0.0.114"
    "15" = "10.0.0.115"
  }
}

resource "aws_security_group" "master" {
    name = "vpc_master"
    description = "Allow incoming HTTP connections."

    # Mesos master
    ingress {
        from_port = 5050
        to_port = 5050
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }
    
    ingress {
        from_port = 5051
        to_port = 5051
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }
    
    # Spark web
    ingress {
        from_port = 4040
        to_port = 4040
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    # Spark master
    ingress {
        from_port = 7077
        to_port = 7077
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }

    # Spark 
    ingress {
        from_port = 7001
        to_port = 7006
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }

    # Jupyter notebook
    ingress {
        from_port = 9999
        to_port = 9999
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }
    ingress {
        from_port = -1
        to_port = -1
        protocol = "icmp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    # Mesos slaves
    egress {
        from_port = 5050
        to_port = 5050
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }
    egress { 
        from_port = 5051
        to_port = 5051
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }

    # Spark master
    egress {
        from_port = 7077
        to_port = 7077
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }

    # Spark 
    egress {
        from_port = 7001
        to_port = 7006
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }

    egress {
        from_port = 4040
        to_port = 4040
        protocol = "tcp"
        cidr_blocks = ["${var.public_subnet_cidr}"]
    }

    egress { 
        from_port = 80
        to_port = 80
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }
    egress { 
        from_port = 443
        to_port = 443
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    vpc_id = "${aws_vpc.default.id}"

    tags {
        Name = "MasterSG"
    }
}


/*
 Public
*/
resource "aws_instance" "mesos-master" {
    instance_type = "m3.large"
    ami = "${var.ami_instance}"
    key_name = "dave"
    security_groups = [ "${aws_security_group.master.id}" ]
    subnet_id = "${aws_subnet.ap-southeast-2-public.id}"
    #private_ip = "10.0.0.55"
    associate_public_ip_address = true
    source_dest_check = false
    connection {
        type = "ssh"
        user = "ubuntu"
        private_key = "${file("${var.aws_key_path}")}"
    }
    provisioner "file" {
        source = "./scripts/"
        destination = "/tmp/"
    }
    provisioner "remote-exec" {
        inline = [ 
            "bash /tmp/docker-install.sh",
            "export MESOS_HOSTNAME=${aws_instance.mesos-master.private_ip}",
            "bash /tmp/run-master.sh",
            "bash /tmp/run-jupyter-notebook.sh ${aws_instance.mesos-master.private_ip}" 
        ]
    }
    tags {
        Name = "revrand-spark-master"
    }
}

resource "aws_eip" "mesos-master" {
    instance = "${aws_instance.mesos-master.id}"
    vpc = true
}

output "mesos-master-ip" {
  value = "${aws_eip.mesos-master.public_ip}"
}

resource "aws_instance" "mesos-slave" {
    instance_type = "m3.large"
    ami = "${var.ami_instance}"
    key_name = "dave"
    security_groups = ["${aws_security_group.master.id}"]
    subnet_id = "${aws_subnet.ap-southeast-2-public.id}"
    #private_ip = "${lookup(slave_ips, count.index)}"
    source_dest_check = false
    connection {
        type = "ssh"
        user = "ubuntu"
        private_key = "${file("${var.aws_key_path}")}"
    }
    provisioner "file" {
        source = "./scripts/"
        destination = "/tmp/"
    }
    provisioner "remote-exec" {
        inline = [ 
            "bash /tmp/docker-install.sh",
            #"export MESOS_HOSTNAME=${lookup(slave_ips, count.index)}",
            "export MESOS_MASTER=${aws_instance.mesos-master.private_ip}:5050",
            "bash /tmp/run-slave.sh" 
        ]
    }

    tags {
        Name = "reverand-spark-slave-${count.index}"
    }

    count = 2
}



