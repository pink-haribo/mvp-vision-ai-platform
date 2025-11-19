"""Check MinIO bucket contents to debug S3 path issue."""
import boto3
from botocore.client import Config

# MinIO connection
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4')
)

bucket = 'training-datasets'
prefix = 'datasets/468cc408-d9cf-47bc-9a0a-d9aaf63b4f35/'

print(f"Checking bucket: {bucket}")
print(f"Prefix: {prefix}")
print("-" * 80)

try:
    # List objects
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=20)

    if 'Contents' in response:
        print(f"Found {len(response['Contents'])} objects:")
        for obj in response['Contents']:
            key = obj['Key']
            size = obj['Size']
            print(f"  - {key} ({size} bytes)")
    else:
        print("No objects found!")

    # Also try without trailing slash
    prefix_no_slash = prefix.rstrip('/')
    print(f"\nTrying without trailing slash: {prefix_no_slash}")
    response2 = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix_no_slash, MaxKeys=20)

    if 'Contents' in response2:
        print(f"Found {len(response2['Contents'])} objects:")
        for obj in response2['Contents'][:5]:
            print(f"  - {obj['Key']}")

except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
