resource "aws_vpc" "default" {
    cidr_block = "${var.vpc_cidr}"
    enable_dns_hostnames = true
    tags {
        Name = "terraform-aws-vpc"
    }
}

resource "aws_internet_gateway" "default" {
    vpc_id = "${aws_vpc.default.id}"
}


/*
    Public Subnet
*/
resource "aws_subnet" "ap-southeast-2-public" {
    vpc_id = "${aws_vpc.default.id}"

    cidr_block = "${var.public_subnet_cidr}"
    availability_zone = "ap-southeast-2a"
    map_public_ip_on_launch = true

    tags {
        Name = "Public Subnet"
    }
}

resource "aws_route_table" "ap-southeast-2-public" {
    vpc_id = "${aws_vpc.default.id}"

    route {
        cidr_block = "0.0.0.0/0"
        gateway_id = "${aws_internet_gateway.default.id}"
    }

    tags {
        Name = "Public Subnet"
    }
}

resource "aws_route_table_association" "ap-southeast-2-public" {
    subnet_id = "${aws_subnet.ap-southeast-2-public.id}"
    route_table_id = "${aws_route_table.ap-southeast-2-public.id}"
}

