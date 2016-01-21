/*
Master
*/

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

    egress { 
        from_port = 0
        to_port = 65535
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
    instance_type = "${var.master_instance}"
    ami = "${var.ami_instance}"
    key_name = "dave"
    security_groups = [ "${aws_security_group.master.id}" ]
    subnet_id = "${aws_subnet.ap-southeast-2-public.id}"
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
            "export TERM=xterm-color",
            "bash /tmp/run-master.sh ${self.private_ip}" ,
            "bash /tmp/run-jupyter-notebook.sh ${self.private_ip}",
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

resource "aws_instance" "mesos-slave" {
    instance_type = "${var.slave_instance}"
    ami = "${var.ami_instance}"
    key_name = "dave"
    security_groups = ["${aws_security_group.master.id}"]
    subnet_id = "${aws_subnet.ap-southeast-2-public.id}"
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
            "bash /tmp/run-slave.sh ${self.private_ip} ${aws_instance.mesos-master.private_ip}", 
        ]
    }

    tags {
        Name = "revrand-spark-slave-${count.index}"
    }

    count = "${var.n_slaves}"
}


output "mesos-master" {
  value = "http://${aws_eip.mesos-master.public_ip}:5050"
}

output "jupyter-notebook" {
  value = "http://${aws_eip.mesos-master.public_ip}:9999"
}

output "spark-master" {
  value = "http://${aws_eip.mesos-master.public_ip}:4040"
}


