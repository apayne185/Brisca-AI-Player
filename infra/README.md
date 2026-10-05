# Infrastructure (AWS CDK)

Deploys the service image to **ECS Fargate** behind an **Application Load
Balancer**:

- VPC across two availability zones with public subnets only. There is no NAT
  gateway, usually the largest fixed cost of a small VPC; tasks get public IPs
  but their security group only admits the load balancer.
- One 1 vCPU / 2 GB task by default, autoscaling to four on CPU (60%) and
  requests per task (300).
- The load balancer routes only to tasks whose `/ready` check passes (agents
  and detector loaded). A deployment circuit breaker rolls back failed releases.
- CloudWatch logs kept for two weeks; alarms on 5xx responses and p95 latency
  above one second, sent to an SNS topic (optional e-mail subscription).

## Use

Requires [uv](https://docs.astral.sh/uv/), Node.js and AWS credentials.

```bash
cd infra
uv sync
uv run pytest                                   # assertions on the synthesized template
npx aws-cdk synth                               # CloudFormation template in cdk.out/
npx aws-cdk deploy -c alarm_email=you@example.com
npx aws-cdk deploy -c image=ghcr.io/apayne185/brisca-ai:0.2.0   # pin a release
npx aws-cdk destroy                             # tear everything down
```

The GHCR package must be public, or the task needs registry credentials.
Running it costs roughly the load balancer plus one Fargate task; `destroy`
removes every resource. The streaming scorer would run as a second Fargate
service against Amazon MSK Serverless; it is not part of this stack.
