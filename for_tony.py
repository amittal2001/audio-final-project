"""
AWS Lambda function for monitoring VPC service quotas
Tracks usage against limits and publishes CloudWatch metrics
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
import time
from datetime import datetime
from concurrent.futures import TimeoutError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure AWS clients with timeouts and retries
aws_config = Config(
    retries=dict(max_attempts=3, mode="adaptive"), read_timeout=30, connect_timeout=30
)


class QuotaError(Exception):
    """Base exception for quota-related errors"""

    pass


class QuotaAPIError(QuotaError):
    """Errors from AWS API calls"""

    pass


class QuotaCalculationError(QuotaError):
    """Errors during quota calculation"""

    pass


def get_aws_client(service_name: str) -> Any:
    """Create AWS client with proper configuration"""
    return boto3.client(service_name, config=aws_config)


def safely_paginate(paginator: Any, operation_name: str, **kwargs) -> List[Dict]:
    """
    Safely handle pagination for AWS API calls with timeout and error handling
    """
    items = []
    try:
        for page in paginator.paginate(**kwargs):
            items.extend(page.get(operation_name, []))
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.error(f"AWS API error during pagination: {error_code} - {str(e)}")
        raise QuotaAPIError(f"Failed to paginate {operation_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during pagination: {str(e)}")
        raise QuotaCalculationError(f"Failed to process {operation_name}: {str(e)}")
    return items


# Usage calculation functions with error handling and pagination
def get_elastic_ip_address_quota_per_nat_gateway() -> int:
    """Get count of EIPs associated with NAT Gateways"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_nat_gateways")
        nat_gateways = safely_paginate(paginator, "NatGateways")

        elastic_ips = set()
        for nat in nat_gateways:
            for address in nat.get("NatGatewayAddresses", []):
                eip_allocation_id = address.get("AllocationId")
                if eip_allocation_id:
                    elastic_ips.add(eip_allocation_id)
        return len(elastic_ips)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count NAT Gateway EIPs: {str(e)}")


def get_inbound_or_outbound_rules_per_security_group() -> int:
    """Get maximum number of rules in any security group"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_security_groups")
        security_groups = safely_paginate(paginator, "SecurityGroups")

        max_rules = 0
        for sg in security_groups:
            inbound_rules = len(sg.get("IpPermissions", []))
            outbound_rules = len(sg.get("IpPermissionsEgress", []))
            max_rules = max(max_rules, inbound_rules + outbound_rules)
        return max_rules
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count security group rules: {str(e)}")


def get_ipv4_cidr_blocks_per_vpc() -> int:
    """Get maximum number of CIDR blocks in any VPC"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpcs")
        vpcs = safely_paginate(paginator, "Vpcs")

        max_cidr_blocks = 0
        for vpc in vpcs:
            num_cidr_blocks = len(vpc.get("CidrBlockAssociationSet", []))
            max_cidr_blocks = max(max_cidr_blocks, num_cidr_blocks)
        return max_cidr_blocks
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count VPC CIDR blocks: {str(e)}")


def get_subnets_per_vpc() -> int:
    """Get maximum number of subnets in any VPC"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpcs")
        vpcs = safely_paginate(paginator, "Vpcs")

        max_subnets = 0
        for vpc in vpcs:
            vpc_id = vpc["VpcId"]
            subnet_paginator = ec2.get_paginator("describe_subnets")
            subnets = safely_paginate(
                subnet_paginator,
                "Subnets",
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}],
            )
            max_subnets = max(max_subnets, len(subnets))
        return max_subnets
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count VPC subnets: {str(e)}")


def get_network_interfaces_per_region() -> int:
    """Get total number of network interfaces in the region"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_network_interfaces")
        interfaces = safely_paginate(paginator, "NetworkInterfaces")
        return len(interfaces)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count network interfaces: {str(e)}")


def get_vpc_security_groups_per_region() -> int:
    """Get total number of security groups in the region"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_security_groups")
        security_groups = safely_paginate(paginator, "SecurityGroups")
        return len(security_groups)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count security groups: {str(e)}")


def get_egress_only_internet_gateways_per_region() -> int:
    """Get total number of egress-only internet gateways"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_egress_only_internet_gateways")
        gateways = safely_paginate(paginator, "EgressOnlyInternetGateways")
        return len(gateways)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count egress-only gateways: {str(e)}")


def get_nat_gateways_per_az() -> int:
    """Get maximum number of NAT gateways in any AZ"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_nat_gateways")
        nat_gateways = safely_paginate(paginator, "NatGateways")

        az_count: Dict[str, int] = {}
        subnet_ids = {nat["SubnetId"] for nat in nat_gateways}

        if subnet_ids:
            try:
                subnet_paginator = ec2.get_paginator("describe_subnets")
                subnets = safely_paginate(
                    subnet_paginator, "Subnets", SubnetIds=list(subnet_ids)
                )
                subnet_az_map = {
                    subnet["SubnetId"]: subnet["AvailabilityZone"] for subnet in subnets
                }

                for nat in nat_gateways:
                    subnet_id = nat["SubnetId"]
                    az = subnet_az_map.get(subnet_id, "Unknown")
                    az_count[az] = az_count.get(az, 0) + 1

                return max(az_count.values()) if az_count else 0

            except ClientError as e:
                if "InvalidSubnetID.NotFound" in str(e):
                    # Handle case where subnets were deleted
                    logger.warning(
                        "Some subnets no longer exist, counting NAT gateways individually"
                    )
                    for nat in nat_gateways:
                        try:
                            subnet = ec2.describe_subnets(SubnetIds=[nat["SubnetId"]])[
                                "Subnets"
                            ][0]
                            az = subnet["AvailabilityZone"]
                            az_count[az] = az_count.get(az, 0) + 1
                        except ClientError:
                            continue
                    return max(az_count.values()) if az_count else 0
                raise
        return 0
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count NAT gateways per AZ: {str(e)}")


def get_active_vpc_peering_connections_per_vpc() -> int:
    """Get maximum number of active peering connections for any VPC"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpc_peering_connections")
        pcx_connections = safely_paginate(paginator, "VpcPeeringConnections")

        vpc_counts: Dict[str, int] = {}
        for pcx in pcx_connections:
            if pcx["Status"]["Code"] == "active":
                requester_vpc = pcx["RequesterVpcInfo"]["VpcId"]
                accepter_vpc = pcx["AccepterVpcInfo"]["VpcId"]
                vpc_counts[requester_vpc] = vpc_counts.get(requester_vpc, 0) + 1
                vpc_counts[accepter_vpc] = vpc_counts.get(accepter_vpc, 0) + 1

        return max(vpc_counts.values()) if vpc_counts else 0
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count peering connections: {str(e)}")


def get_vpc_peering_expiry_hours() -> float:
    """Get minimum remaining hours for pending VPC peering requests"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpc_peering_connections")
        filters = [{"Name": "status-code", "Values": ["pending-acceptance"]}]
        pcx_connections = safely_paginate(
            paginator, "VpcPeeringConnections", Filters=filters
        )

        if not pcx_connections:
            return 0

        now = datetime.utcnow()
        min_remaining_hours = float("inf")

        for pcx in pcx_connections:
            expiry_time = pcx["ExpirationTime"]
            remaining_time = expiry_time - now
            remaining_hours = remaining_time.total_seconds() / 3600
            min_remaining_hours = min(min_remaining_hours, remaining_hours)

        return min_remaining_hours if min_remaining_hours != float("inf") else 0
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to calculate peering expiry: {str(e)}")


def get_private_ip_address_quota_per_nat_gateway() -> int:
    """Get total number of private IPs associated with NAT Gateways"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_nat_gateways")
        nat_gateways = safely_paginate(paginator, "NatGateways")

        total_private_ips = 0
        for nat in nat_gateways:
            for address in nat.get("NatGatewayAddresses", []):
                network_interface_id = address.get("NetworkInterfaceId")
                if network_interface_id:
                    try:
                        eni_paginator = ec2.get_paginator("describe_network_interfaces")
                        interfaces = safely_paginate(
                            eni_paginator,
                            "NetworkInterfaces",
                            NetworkInterfaceIds=[network_interface_id],
                        )
                        if interfaces:
                            total_private_ips += len(
                                interfaces[0].get("PrivateIpAddresses", [])
                            )
                    except ClientError as e:
                        if "InvalidNetworkInterfaceID.NotFound" in str(e):
                            logger.warning(
                                f"Network interface {network_interface_id} no longer exists"
                            )
                            continue
                        raise
        return total_private_ips
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(
            f"Failed to count NAT Gateway private IPs: {str(e)}"
        )


def get_internet_gateways_per_region() -> int:
    """Get total number of internet gateways in the region"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_internet_gateways")
        gateways = safely_paginate(paginator, "InternetGateways")
        return len(gateways)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count internet gateways: {str(e)}")


def get_interface_vpc_endpoints_per_vpc() -> int:
    """Get maximum number of interface endpoints in any VPC"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpc_endpoints")
        endpoints = safely_paginate(
            paginator,
            "VpcEndpoints",
            Filters=[{"Name": "vpc-endpoint-type", "Values": ["Interface"]}],
        )

        vpc_counts: Dict[str, int] = {}
        for endpoint in endpoints:
            vpc_id = endpoint["VpcId"]
            vpc_counts[vpc_id] = vpc_counts.get(vpc_id, 0) + 1

        return max(vpc_counts.values()) if vpc_counts else 0
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(
            f"Failed to count interface endpoints per VPC: {str(e)}"
        )


def get_route_tables_per_vpc() -> int:
    """Get maximum number of route tables in any VPC"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpcs")
        vpcs = safely_paginate(paginator, "Vpcs")

        max_route_tables = 0
        for vpc in vpcs:
            vpc_id = vpc["VpcId"]
            rt_paginator = ec2.get_paginator("describe_route_tables")
            route_tables = safely_paginate(
                rt_paginator,
                "RouteTables",
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}],
            )
            max_route_tables = max(max_route_tables, len(route_tables))

        return max_route_tables
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count route tables per VPC: {str(e)}")


def get_gateway_vpc_endpoints_per_region() -> int:
    """Get total number of gateway endpoints in the region"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpc_endpoints")
        endpoints = safely_paginate(
            paginator,
            "VpcEndpoints",
            Filters=[{"Name": "vpc-endpoint-type", "Values": ["Gateway"]}],
        )
        return len(endpoints)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count gateway endpoints: {str(e)}")


def get_routes_per_route_table() -> int:
    """Get maximum number of routes in any route table"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_route_tables")
        route_tables = safely_paginate(paginator, "RouteTables")

        max_routes = 0
        for rt in route_tables:
            num_routes = len(rt.get("Routes", []))
            max_routes = max(max_routes, num_routes)

        return max_routes
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count routes per route table: {str(e)}")


def get_vpcs_per_region() -> int:
    """Get total number of VPCs in the region"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpcs")
        vpcs = safely_paginate(paginator, "Vpcs")
        return len(vpcs)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count VPCs in region: {str(e)}")


def get_network_acls_per_vpc() -> int:
    """Get maximum number of network ACLs in any VPC"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpcs")
        vpcs = safely_paginate(paginator, "Vpcs")

        max_acls = 0
        for vpc in vpcs:
            vpc_id = vpc["VpcId"]
            nacl_paginator = ec2.get_paginator("describe_network_acls")
            network_acls = safely_paginate(
                nacl_paginator,
                "NetworkAcls",
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}],
            )
            max_acls = max(max_acls, len(network_acls))

        return max_acls
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count network ACLs per VPC: {str(e)}")


def get_rules_per_network_acl() -> int:
    """Get maximum number of rules in any network ACL"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_network_acls")
        network_acls = safely_paginate(paginator, "NetworkAcls")

        max_rules = 0
        for nacl in network_acls:
            num_entries = len(nacl.get("Entries", []))
            max_rules = max(max_rules, num_entries)

        return max_rules
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count rules per network ACL: {str(e)}")


def get_security_groups_per_network_interface() -> int:
    """Get maximum number of security groups attached to any network interface"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_network_interfaces")
        interfaces = safely_paginate(paginator, "NetworkInterfaces")

        max_sgs = 0
        for ni in interfaces:
            num_sgs = len(ni.get("Groups", []))
            max_sgs = max(max_sgs, num_sgs)

        return max_sgs
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(
            f"Failed to count security groups per interface: {str(e)}"
        )


def get_subnets_shared_with_account() -> int:
    """Get count of subnets shared with this account"""
    ram_client = get_aws_client("ram")
    try:
        shared_subnet_count = 0
        paginator = ram_client.get_paginator("get_resource_shares")

        # Get resource shares where we are the recipient
        shares = safely_paginate(
            paginator, "resourceShares", resourceOwner="OTHER-ACCOUNTS"
        )

        for share in shares:
            try:
                # Get resources for each share
                resources = ram_client.list_resources(
                    resourceOwner="SELF",
                    resourceShareArns=[share["resourceShareArn"]],
                    resourceType="ec2:Subnet",
                )
                shared_subnet_count += len(resources.get("resources", []))
            except ClientError as e:
                logger.warning(
                    f"Error getting resources for share {share['resourceShareArn']}: {str(e)}"
                )
                continue

        return shared_subnet_count
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count shared subnets: {str(e)}")


def get_elastic_ips_per_region() -> int:
    """Get total number of Elastic IPs in the region"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_addresses")
        addresses = safely_paginate(paginator, "Addresses")
        return len(addresses)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to count Elastic IPs: {str(e)}")


def get_outstanding_vpc_peering_requests() -> int:
    """Get number of outstanding VPC peering connection requests"""
    ec2 = get_aws_client("ec2")
    try:
        paginator = ec2.get_paginator("describe_vpc_peering_connections")
        pending_connections = safely_paginate(
            paginator,
            "VpcPeeringConnections",
            Filters=[{"Name": "status-code", "Values": ["pending-acceptance"]}],
        )
        return len(pending_connections)
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(
            f"Failed to count pending peering requests: {str(e)}"
        )


def get_participant_accounts_per_vpc() -> int:
    """Get maximum number of participant accounts for any VPC"""
    ram_client = get_aws_client("ram")
    ec2 = get_aws_client("ec2")
    try:
        # Get all VPCs first
        vpc_paginator = ec2.get_paginator("describe_vpcs")
        vpcs = safely_paginate(vpc_paginator, "Vpcs")
        max_participants = 0

        # For each VPC, check RAM shares
        for vpc in vpcs:
            try:
                # Get resource shares for this VPC
                share_paginator = ram_client.get_paginator("get_resource_shares")
                shares = safely_paginate(
                    share_paginator,
                    "resourceShares",
                    resourceOwner="SELF",
                    resourceShareStatus="ACTIVE",
                )

                # Count unique principal accounts for this VPC
                vpc_participants = set()
                for share in shares:
                    # Get principals (accounts) for this share
                    principals = ram_client.list_principals(
                        resourceOwner="SELF",
                        resourceShareArns=[share["resourceShareArn"]],
                    )

                    # Add each principal to our set
                    for principal in principals.get("principals", []):
                        vpc_participants.add(principal["id"])

                max_participants = max(max_participants, len(vpc_participants))

            except ClientError as e:
                logger.warning(
                    f"Error checking RAM shares for VPC {vpc['VpcId']}: {str(e)}"
                )
                continue

        return max_participants
    except (QuotaAPIError, QuotaCalculationError):
        raise
    except Exception as e:
        raise QuotaCalculationError(
            f"Failed to count participant accounts per VPC: {str(e)}"
        )


### USAGE_FUNCTIONS dictionary ###

USAGE_FUNCTIONS = {
    "L-5F53652F": {
        "QuotaName": "Elastic IP address quota per NAT gateway",
        "MetricName": "ElasticIPAddressQuotaPerNATGateway",
        "UsageFunction": get_elastic_ip_address_quota_per_nat_gateway,
    },
    "L-0EA8095F": {
        "QuotaName": "Inbound or outbound rules per security group",
        "MetricName": "InboundOutboundRulesPerSecurityGroup",
        "UsageFunction": get_inbound_or_outbound_rules_per_security_group,
    },
    "L-83CA0A9D": {
        "QuotaName": "IPv4 CIDR blocks per VPC",
        "MetricName": "IPv4CIDRBlocksPerVPC",
        "UsageFunction": get_ipv4_cidr_blocks_per_vpc,
    },
    "L-407747CB": {
        "QuotaName": "Subnets per VPC",
        "MetricName": "SubnetsPerVPC",
        "UsageFunction": get_subnets_per_vpc,
    },
    "L-2C462E13": {
        "QuotaName": "Participant accounts per VPC",
        "MetricName": "ParticipantAccountsPerVPC",
        "UsageFunction": get_participant_accounts_per_vpc,
    },
    "L-DF5E4CA3": {
        "QuotaName": "Network interfaces per Region",
        "MetricName": "NetworkInterfacesPerRegion",
        "UsageFunction": get_network_interfaces_per_region,
    },
    "L-E79EC296": {
        "QuotaName": "VPC security groups per Region",
        "MetricName": "VPCSecurityGroupsPerRegion",
        "UsageFunction": get_vpc_security_groups_per_region,
    },
    "L-45FE3B85": {
        "QuotaName": "Egress-only internet gateways per Region",
        "MetricName": "EgressOnlyInternetGatewaysPerRegion",
        "UsageFunction": get_egress_only_internet_gateways_per_region,
    },
    "L-DFA99DE7": {
        "QuotaName": "Private IP address quota per NAT gateway",
        "MetricName": "PrivateIPAddressQuotaPerNATGateway",
        "UsageFunction": get_private_ip_address_quota_per_nat_gateway,
    },
    "L-FE5A380F": {
        "QuotaName": "NAT gateways per Availability Zone",
        "MetricName": "NATGatewaysPerAZ",
        "UsageFunction": get_nat_gateways_per_az,
    },
    "L-7E9ECCDB": {
        "QuotaName": "Active VPC peering connections per VPC",
        "MetricName": "ActiveVPCPeeringConnectionsPerVPC",
        "UsageFunction": get_active_vpc_peering_connections_per_vpc,
    },
    "L-A4707A72": {
        "QuotaName": "Internet gateways per Region",
        "MetricName": "InternetGatewaysPerRegion",
        "UsageFunction": get_internet_gateways_per_region,
    },
    "L-8312C5BB": {
        "QuotaName": "VPC peering connection request expiry hours",
        "MetricName": "VPCPeeringConnectionRequestExpiryHours",
        "UsageFunction": get_vpc_peering_expiry_hours,
    },
    "L-29B6F2EB": {
        "QuotaName": "Interface VPC endpoints per VPC",
        "MetricName": "InterfaceVPCEndpointsPerVPC",
        "UsageFunction": get_interface_vpc_endpoints_per_vpc,
    },
    "L-589F43AA": {
        "QuotaName": "Route tables per VPC",
        "MetricName": "RouteTablesPerVPC",
        "UsageFunction": get_route_tables_per_vpc,
    },
    "L-1B52E74A": {
        "QuotaName": "Gateway VPC endpoints per Region",
        "MetricName": "GatewayVPCEndpointsPerRegion",
        "UsageFunction": get_gateway_vpc_endpoints_per_region,
    },
    "L-93826ACB": {
        "QuotaName": "Routes per route table",
        "MetricName": "RoutesPerRouteTable",
        "UsageFunction": get_routes_per_route_table,
    },
    "L-F678F1CE": {
        "QuotaName": "VPCs per Region",
        "MetricName": "VPCsPerRegion",
        "UsageFunction": get_vpcs_per_region,
    },
    "L-B4A6D682": {
        "QuotaName": "Network ACLs per VPC",
        "MetricName": "NetworkACLsPerVPC",
        "UsageFunction": get_network_acls_per_vpc,
    },
    "L-2AEEBF1A": {
        "QuotaName": "Rules per network ACL",
        "MetricName": "RulesPerNetworkACL",
        "UsageFunction": get_rules_per_network_acl,
    },
    "L-DC9F7029": {
        "QuotaName": "Outstanding VPC peering connection requests",
        "MetricName": "OutstandingVPCPeeringRequests",
        "UsageFunction": get_outstanding_vpc_peering_requests,
    },
    "L-2AFB9258": {
        "QuotaName": "Security groups per network interface",
        "MetricName": "SecurityGroupsPerNetworkInterface",
        "UsageFunction": get_security_groups_per_network_interface,
    },
    "L-44499CD2": {
        "QuotaName": "Subnets that can be shared with an account",
        "MetricName": "SubnetsSharedWithAccount",
        "UsageFunction": get_subnets_shared_with_account,
    },
}


def calculate_quota_usage(quota_info: Dict) -> float:
    """
    Calculate quota usage with proper error handling
    Returns: Raw usage value
    """
    quota_code = quota_info["QuotaCode"]

    if quota_code not in USAGE_FUNCTIONS:
        logger.info(f"No usage function defined for quota {quota_code}")
        return 0

    try:
        usage = USAGE_FUNCTIONS[quota_code]["UsageFunction"]()
        logger.debug(f"Quota {quota_info['QuotaName']}: Raw usage={usage}")
        return usage

    except (QuotaAPIError, QuotaCalculationError) as e:
        logger.error(f"Failed to calculate usage for {quota_code}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calculating usage for {quota_code}: {str(e)}")
        raise QuotaCalculationError(f"Failed to calculate quota usage: {str(e)}")


def publish_metrics(service_code: str, quota_info: Dict, usage: float,aws_account_id: str) -> None:
    """
    Publish metrics to CloudWatch with error handling.
    Publishes raw usage and quota values separately.
    """
    cloudwatch = get_aws_client("cloudwatch")
    metric_name = USAGE_FUNCTIONS[quota_info["QuotaCode"]]["MetricName"]
    quota_value = quota_info["Value"]

    try:
        metric_data = [
            {
                "MetricName": f"{metric_name}_Usage",
                "Dimensions": [
                    {"Name": "Service", "Value": service_code},
                    {"Name": "QuotaName", "Value": quota_info["QuotaName"]},
                    {"Name":"accountID","Value":aws_account_id},
                ],
                "Value": usage,
            },
            {
                "MetricName": f"{metric_name}_Quota",
                "Dimensions": [
                    {"Name": "Service", "Value": service_code},
                    {"Name": "QuotaName", "Value": quota_info["QuotaName"]},
                    {"Name":"accountID","Value":aws_account_id},
                ],
                "Value": quota_value,
            },
        ]

        cloudwatch.put_metric_data(Namespace="RawServiceQuotas", MetricData=metric_data)

        logger.info(
            f"Published metrics for {quota_info['QuotaName']}: "
            f"Usage={usage}, Quota={quota_value}"
        )

    except ClientError as e:
        logger.error(f"Failed to publish CloudWatch metrics: {str(e)}")
        raise QuotaAPIError(f"Failed to publish metrics: {str(e)}")


def get_service_quotas(service_code: str) -> List[Dict]:
    """Get service quotas with pagination and error handling"""
    sq_client = get_aws_client("service-quotas")
    try:
        paginator = sq_client.get_paginator("list_service_quotas")
        return safely_paginate(paginator, "Quotas", ServiceCode=service_code)
    except (QuotaAPIError, QuotaCalculationError) as e:
        logger.error(f"Failed to get service quotas: {str(e)}")
        raise
    except Exception as e:
        raise QuotaCalculationError(f"Failed to retrieve service quotas: {str(e)}")


def process_single_quota(quota: Dict, service_code: str,aws_account_id: str) -> Tuple[bool, str]:
    """
    Process a single quota and publish its metrics
    Returns: Tuple of (success: bool, error_message: str)
    """
    quota_code = quota["QuotaCode"]
    quota_name = quota["QuotaName"]

    try:
        # Skip if we don't have a usage function
        if quota_code not in USAGE_FUNCTIONS:
            return True, ""

        # Calculate usage
        usage = calculate_quota_usage(quota)

        # Publish metrics
        publish_metrics(service_code, quota, usage,aws_account_id)

        logger.info(
            f"Processed quota {quota_name}: " f"Usage={usage}, Limit={quota['Value']}"
        )
        return True, ""

    except (QuotaAPIError, QuotaCalculationError) as e:
        error_msg = f"Failed to process quota {quota_name}: {str(e)}"
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error processing quota {quota_name}: {str(e)}"
        return False, error_msg


def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    Main Lambda handler with proper error handling and logging

    Args:
        event: Lambda event data
        context: Lambda context object

    Returns:
        Dict containing execution statistics and status
    """
    service_code = "vpc"
    start_time = time.time()
    logger.info(f"Starting VPC quota check at {start_time}")
    try:
        sts_client = boto3.client('sts')
        aws_account_id = sts_client.get_caller_identity()['Account']
    except Exception as e:
        logger.error(f"failed to connect to sts client with error {str(e)}")
        return {"statusCode": 500,"body": "failed to connect to sts client"}
    try:
        # Get all quotas
        quotas = get_service_quotas(service_code)
        quota_count = len(quotas)
        logger.info(f"Found {quota_count} quotas to process")

        # Initialize tracking
        processed_count = 0
        error_count = 0
        errors = []

        # Process each quota
        for quota in quotas:
            success, error_msg = process_single_quota(quota, service_code,aws_account_id)

            if success:
                processed_count += 1
            else:
                error_count += 1
                errors.append(error_msg)

        # Calculate execution statistics
        execution_time = time.time() - start_time
        success_rate = (processed_count / quota_count * 100) if quota_count > 0 else 0

        # Log completion summary
        logger.info(
            f"Completed processing {processed_count} quotas with {error_count} "
            f"errors in {execution_time:.1f} seconds. "
            f"Success rate: {success_rate:.1f}%"
        )

        if errors:
            logger.warning("Errors encountered during execution:")
            for error in errors:
                logger.warning(error)

        # Return execution results
        return {
            "statusCode": 200,
            "body": {
                "quotas_processed": processed_count,
                "quotas_total": quota_count,
                "errors": error_count,
                "success_rate": success_rate,
                "execution_time": execution_time,
                "error_messages": errors if errors else None,
            },
        }

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Critical error in Lambda execution: {str(e)}"
        logger.error(error_msg)

        return {
            "statusCode": 500,
            "body": {"error": error_msg, "execution_time": execution_time},
        }


if __name__ == "__main__":
    # For local testing
    test_event = {}
    test_context = None
    result = lambda_handler(test_event, test_context)
    print(result)
