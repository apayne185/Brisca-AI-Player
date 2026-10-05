"""CDK entry point: ``cdk synth`` / ``cdk deploy`` (see infra/README.md)."""

import os

import aws_cdk as cdk

from brisca_infra.stack import DEFAULT_IMAGE, BriscaServiceStack

app = cdk.App()
BriscaServiceStack(
    app,
    "BriscaAi",
    image=app.node.try_get_context("image") or DEFAULT_IMAGE,
    alarm_email=app.node.try_get_context("alarm_email"),
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION", "eu-west-1"),
    ),
)
app.synth()
