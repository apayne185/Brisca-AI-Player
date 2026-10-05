"""ECS Fargate deployment of the Brisca AI service behind a load balancer.

Cost-conscious by design: public subnets only (no NAT gateway, the largest fixed
cost of a small VPC), one task by default with autoscaling for load, and
two-week log retention. Tasks accept traffic only from the load balancer.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import CfnOutput, Duration, Stack, Tags
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from constructs import Construct

DEFAULT_IMAGE = "ghcr.io/apayne185/brisca-ai:main"
CONTAINER_PORT = 8000


class BriscaServiceStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        image: str = DEFAULT_IMAGE,
        desired_count: int = 1,
        max_count: int = 4,
        alarm_email: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                )
            ],
        )
        cluster = ecs.Cluster(
            self, "Cluster", vpc=vpc, container_insights_v2=ecs.ContainerInsights.ENABLED
        )

        self.service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Service",
            cluster=cluster,
            cpu=1024,
            memory_limit_mib=2048,
            desired_count=desired_count,
            public_load_balancer=True,
            assign_public_ip=True,  # public subnets, no NAT
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            health_check_grace_period=Duration.seconds(60),
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_registry(image),
                container_port=CONTAINER_PORT,
                environment={"BRISCA_MODELS_DIR": "/app/models"},
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="api", log_retention=logs.RetentionDays.TWO_WEEKS
                ),
            ),
        )
        # Route traffic only to tasks whose agents and detector have loaded.
        self.service.target_group.configure_health_check(
            path="/ready",
            healthy_http_codes="200",
            interval=Duration.seconds(15),
            healthy_threshold_count=2,
        )

        scaling = self.service.service.auto_scale_task_count(
            min_capacity=desired_count, max_capacity=max_count
        )
        scaling.scale_on_cpu_utilization("CpuScaling", target_utilization_percent=60)
        scaling.scale_on_request_count(
            "RequestScaling",
            requests_per_target=300,
            target_group=self.service.target_group,
        )

        self._alarms(alarm_email)
        Tags.of(self).add("project", "brisca-ai")
        CfnOutput(
            self,
            "ServiceUrl",
            value=f"http://{self.service.load_balancer.load_balancer_dns_name}",
        )

    def _alarms(self, email: str | None) -> None:
        topic = sns.Topic(self, "AlarmTopic", display_name="Brisca AI alarms")
        if email:
            topic.add_subscription(subscriptions.EmailSubscription(email))
        metrics = self.service.load_balancer.metrics
        alarms = [
            cloudwatch.Alarm(
                self,
                "Target5xx",
                alarm_description="Tasks are returning server errors",
                metric=metrics.http_code_target(
                    code=elbv2.HttpCodeTarget.TARGET_5XX_COUNT, period=Duration.minutes(5)
                ),
                threshold=5,
                evaluation_periods=1,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
            cloudwatch.Alarm(
                self,
                "LatencyP95",
                alarm_description="p95 response time above one second",
                metric=metrics.target_response_time(statistic="p95", period=Duration.minutes(5)),
                threshold=1.0,
                evaluation_periods=3,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
        ]
        for alarm in alarms:
            alarm.add_alarm_action(cw_actions.SnsAction(topic))
