"""Assertions on the synthesized CloudFormation template; nothing is deployed."""

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from brisca_infra.stack import BriscaServiceStack


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App()
    stack = BriscaServiceStack(app, "Test", image="example/brisca:1", alarm_email="ops@example.com")
    return Template.from_stack(stack)


def test_no_nat_gateways(template: Template) -> None:
    template.resource_count_is("AWS::EC2::NatGateway", 0)


def test_container_runs_the_image_on_port_8000(template: Template) -> None:
    template.has_resource_properties(
        "AWS::ECS::TaskDefinition",
        {
            "Cpu": "1024",
            "Memory": "2048",
            "ContainerDefinitions": [
                Match.object_like(
                    {
                        "Image": "example/brisca:1",
                        "PortMappings": [Match.object_like({"ContainerPort": 8000})],
                        "Environment": [{"Name": "BRISCA_MODELS_DIR", "Value": "/app/models"}],
                    }
                )
            ],
        },
    )


def test_load_balancer_checks_readiness(template: Template) -> None:
    template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::TargetGroup",
        {"HealthCheckPath": "/ready", "Port": 80, "Protocol": "HTTP"},
    )
    template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::Listener", {"Port": 80, "Protocol": "HTTP"}
    )


def test_deployments_roll_back_on_failure(template: Template) -> None:
    template.has_resource_properties(
        "AWS::ECS::Service",
        {
            "DeploymentConfiguration": Match.object_like(
                {"DeploymentCircuitBreaker": {"Enable": True, "Rollback": True}}
            ),
            "LaunchType": "FARGATE",
        },
    )


def test_tasks_only_accept_traffic_from_the_load_balancer(template: Template) -> None:
    ingress = template.find_resources("AWS::EC2::SecurityGroupIngress")
    to_tasks = [r for r in ingress.values() if r["Properties"].get("FromPort") == 8000]
    assert to_tasks
    assert all("SourceSecurityGroupId" in r["Properties"] for r in to_tasks)


def test_autoscaling_on_cpu_and_requests(template: Template) -> None:
    template.has_resource_properties(
        "AWS::ApplicationAutoScaling::ScalableTarget", {"MinCapacity": 1, "MaxCapacity": 4}
    )
    template.resource_count_is("AWS::ApplicationAutoScaling::ScalingPolicy", 2)


def test_alarms_notify_by_email(template: Template) -> None:
    template.resource_count_is("AWS::CloudWatch::Alarm", 2)
    template.has_resource_properties(
        "AWS::SNS::Subscription", {"Protocol": "email", "Endpoint": "ops@example.com"}
    )


def test_logs_are_retained_for_two_weeks(template: Template) -> None:
    template.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 14})
