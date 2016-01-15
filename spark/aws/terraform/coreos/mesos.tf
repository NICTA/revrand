/*
Master
*/

resource "aws_security_group" "master" {
    name = "vpc_master"
    description = "Allow incoming HTTP connections."

    #etcd
    ingress {
        from_port = 2379
        to_port = 2380
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }
 
    ingress {
        from_port = 4001
        to_port = 4001
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    ingress {
        from_port = 7001
        to_port = 7001
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }


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
    instance_type = "m3.large"
    ami = "${var.ami_instance}"
    key_name = "dave"
    security_groups = [ "${aws_security_group.master.id}" ]
    subnet_id = "${aws_subnet.ap-southeast-2-public.id}"
    associate_public_ip_address = true
    source_dest_check = false
    #user_data = "${template_file.cloud_config.rendered}"
    user_data = "${file("./cloud_config.yaml")}"
    connection {
        type = "ssh"
        user = "core"
        private_key = "${file("${var.aws_key_path}")}"
    }
    provisioner "file" {
        source = "./scripts/"
        destination = "/tmp/"
    }

    provisioner "remote-exec" {
        inline = [ 
            "export TERM=xterm-color",
            "bash /tmp/run-master.sh",
        ]
    }
  
    tags {
        Name = "revrand-spark-master"
    }
}

resource "template_file" "cloud_config" {
    depends_on = [
        "template_file.etcd_discovery_url"
    ]
    template = "${file("./cloud_config.yaml.tpl")}"
    vars {
        etcd_discovery_url = "${file(var.ETCD_DISCOVERY_URL_FILE)}"
    }
}


resource "template_file" "etcd_discovery_url" {
    template = "/dev/null"
    provisioner "local-exec" {
        command = "curl https://discovery.etcd.io/new?size=3 > ${var.ETCD_DISCOVERY_URL_FILE}"
    }
}


variable "ETCD_COUNT" {
    default = 5
}
variable "ETCD_DISCOVERY_URL_FILE" {
    default = "etcd_discovery_url.txt"
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
    user_data = "${file("./cloud_config.yaml")}"
    connection {
        type = "ssh"
        user = "core"
        private_key = "${file("${var.aws_key_path}")}"
    }
    provisioner "file" {
        source = "./scripts/"
        destination = "/tmp/"
    }

    provisioner "remote-exec" {
        inline = [ 
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



