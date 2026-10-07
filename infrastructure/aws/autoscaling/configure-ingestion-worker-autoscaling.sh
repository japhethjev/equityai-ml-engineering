#!/usr/bin/env bash
set -euo pipefail

REGION="eu-west-2"
CLUSTER="equityai-production"
SERVICE="equityai-ingestion-worker-service-0s739p5o"
RESOURCE_ID="service/${CLUSTER}/${SERVICE}"
QUEUE_NAME="equityai-document-ingestion"

MIN_CAPACITY=1
MAX_CAPACITY=5

SCALE_OUT_POLICY="equityai-ingestion-worker-scale-out"
SCALE_IN_POLICY="equityai-ingestion-worker-scale-in"

SCALE_OUT_ALARM="equityai-ingestion-worker-backlog-high"
SCALE_IN_ALARM="equityai-ingestion-worker-idle"

echo "Registering ECS scalable target..."

aws application-autoscaling register-scalable-target \
  --region "$REGION" \
  --service-namespace ecs \
  --resource-id "$RESOURCE_ID" \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity "$MIN_CAPACITY" \
  --max-capacity "$MAX_CAPACITY"

echo "Creating scale-out policy..."

SCALE_OUT_ARN=$(aws application-autoscaling put-scaling-policy \
  --region "$REGION" \
  --service-namespace ecs \
  --resource-id "$RESOURCE_ID" \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name "$SCALE_OUT_POLICY" \
  --policy-type StepScaling \
  --step-scaling-policy-configuration '{
    "AdjustmentType": "ChangeInCapacity",
    "Cooldown": 300,
    "MetricAggregationType": "Maximum",
    "StepAdjustments": [
      {
        "MetricIntervalLowerBound": 0,
        "MetricIntervalUpperBound": 5,
        "ScalingAdjustment": 1
      },
      {
        "MetricIntervalLowerBound": 5,
        "MetricIntervalUpperBound": 20,
        "ScalingAdjustment": 2
      },
      {
        "MetricIntervalLowerBound": 20,
        "ScalingAdjustment": 4
      }
    ]
  }' \
  --query 'PolicyARN' \
  --output text)

echo "Creating scale-in policy..."

SCALE_IN_ARN=$(aws application-autoscaling put-scaling-policy \
  --region "$REGION" \
  --service-namespace ecs \
  --resource-id "$RESOURCE_ID" \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name "$SCALE_IN_POLICY" \
  --policy-type StepScaling \
  --step-scaling-policy-configuration '{
    "AdjustmentType": "ChangeInCapacity",
    "Cooldown": 600,
    "MetricAggregationType": "Maximum",
    "StepAdjustments": [
      {
        "MetricIntervalUpperBound": 0,
        "ScalingAdjustment": -1
      }
    ]
  }' \
  --query 'PolicyARN' \
  --output text)

echo "Creating backlog scale-out alarm..."

aws cloudwatch put-metric-alarm \
  --region "$REGION" \
  --alarm-name "$SCALE_OUT_ALARM" \
  --alarm-description "Scale out EquityAI ingestion workers when financial-report backlog builds." \
  --namespace AWS/SQS \
  --metric-name ApproximateNumberOfMessagesVisible \
  --dimensions Name=QueueName,Value="$QUEUE_NAME" \
  --statistic Maximum \
  --period 60 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$SCALE_OUT_ARN"


echo "Creating idle scale-in alarm..."

aws cloudwatch put-metric-alarm \
  --region "$REGION" \
  --alarm-name "$SCALE_IN_ALARM" \
  --alarm-description "Scale in only when no queued or in-flight ingestion messages remain." \
  --evaluation-periods 2 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data notBreaching \
  --metrics "[
    {
      \"Id\": \"visible\",
      \"MetricStat\": {
        \"Metric\": {
          \"Namespace\": \"AWS/SQS\",
          \"MetricName\": \"ApproximateNumberOfMessagesVisible\",
          \"Dimensions\": [
            {
              \"Name\": \"QueueName\",
              \"Value\": \"$QUEUE_NAME\"
            }
          ]
        },
        \"Period\": 300,
        \"Stat\": \"Maximum\"
      },
      \"ReturnData\": false
    },
    {
      \"Id\": \"inflight\",
      \"MetricStat\": {
        \"Metric\": {
          \"Namespace\": \"AWS/SQS\",
          \"MetricName\": \"ApproximateNumberOfMessagesNotVisible\",
          \"Dimensions\": [
            {
              \"Name\": \"QueueName\",
              \"Value\": \"$QUEUE_NAME\"
            }
          ]
        },
        \"Period\": 300,
        \"Stat\": \"Maximum\"
      },
      \"ReturnData\": false
    },
    {
      \"Id\": \"workremaining\",
      \"Expression\": \"visible + inflight\",
      \"Label\": \"Outstanding ingestion messages\",
      \"ReturnData\": true
    }
  ]" \
  --alarm-actions "$SCALE_IN_ARN"

echo "EquityAI ingestion-worker autoscaling configured."
