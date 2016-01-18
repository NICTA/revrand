variable "aws_access_key" {}
variable "aws_secret_key" {}
variable "aws_key_name" {}
variable "aws_key_path" {}

variable "region" {
    default = "ap-southeast-2"
}

variable "master_instance" {
    default = "m3.large"
}

variable "slave_instance" {
    default = "m3.large"
}

variable "n_slaves" {
    default = "3"
}

variable "ami_instance" {
    default = "ami-d2e8cdb1" # Build by Packer (mesos + docker installed)
}

variable "vpc_cidr" {
    description = "CIDR for the whole VPC"
    default = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
    description = "CIDR for the Public Subnet"
    default = "10.0.0.0/24"
}

variable "private_subnet_cidr" {
    description = "CIDR for the Private Subnet"
    default = "10.0.1.0/24"
}
