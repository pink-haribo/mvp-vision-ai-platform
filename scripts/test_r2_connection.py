"""
Test Cloudflare R2 Connection

This script tests the connection to Cloudflare R2 (S3-compatible storage).
Used for verifying R2 credentials before deploying MLflow.

Usage:
    python scripts/test_r2_connection.py
"""

import os
import sys
from pathlib import Path

try:
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("[!] boto3 not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError, NoCredentialsError


def load_env_file(env_file_path):
    """Load environment variables from .env file."""
    if not os.path.exists(env_file_path):
        return False

    print(f"[FILE] Loading environment variables from {env_file_path}")
    loaded_count = 0

    with open(env_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Set environment variable if not already set
                if not os.getenv(key):
                    os.environ[key] = value
                    loaded_count += 1

    print(f"[OK] Loaded {loaded_count} environment variables")
    return True


def test_r2_connection():
    """Test R2 connection with credentials from environment variables."""

    print("=" * 60)
    print("Cloudflare R2 Connection Test")
    print("=" * 60)

    # Try to load from .env.r2 file first
    project_root = Path(__file__).parent.parent
    env_r2_path = project_root / ".env.r2"

    if env_r2_path.exists():
        load_env_file(env_r2_path)

    # Get credentials from environment
    r2_endpoint = os.getenv("AWS_S3_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME", "vision-platform-prod")

    # Validate credentials
    if not all([r2_endpoint, access_key, secret_key]):
        print("\n[ERROR] Missing R2 credentials!")
        print("\nPlease set the following environment variables:")
        print("  - AWS_S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com")
        print("  - AWS_ACCESS_KEY_ID=<your_access_key>")
        print("  - AWS_SECRET_ACCESS_KEY=<your_secret_key>")
        print("  - R2_BUCKET_NAME=vision-platform-prod (optional)")
        print("\nExample:")
        print('  $env:AWS_S3_ENDPOINT_URL="https://a1b2c3d4.r2.cloudflarestorage.com"')
        print('  $env:AWS_ACCESS_KEY_ID="1234567890abcdef"')
        print('  $env:AWS_SECRET_ACCESS_KEY="abcdef1234567890"')
        print('  python scripts/test_r2_connection.py')
        return False

    print(f"\n[INFO] Configuration:")
    print(f"  Endpoint: {r2_endpoint}")
    print(f"  Access Key: {access_key[:8]}...{access_key[-4:]}")
    print(f"  Secret Key: {'*' * 20}")
    print(f"  Bucket: {bucket_name}")

    try:
        # Create S3 client
        print("\n[CONNECT] Creating S3 client...")
        s3 = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4')
        )
        print("[OK] S3 client created successfully")

        # Test 1: List buckets
        print("\n[BUCKET] Test 1: Listing buckets...")
        response = s3.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        print(f"[OK] Found {len(buckets)} bucket(s):")
        for b in buckets:
            print(f"  - {b}")

        if bucket_name not in buckets:
            print(f"\n[WARN]  Warning: Bucket '{bucket_name}' not found!")
            print("   Please create the bucket in Cloudflare R2 Dashboard")
            return False

        # Test 2: Upload file
        print(f"\n[UPLOAD] Test 2: Uploading test file to '{bucket_name}'...")
        test_content = b"Hello from Vision AI Platform!\nThis is a test file for MLflow artifacts."
        test_key = "test/connection_test.txt"

        s3.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content,
            ContentType='text/plain'
        )
        print(f"[OK] Upload successful: s3://{bucket_name}/{test_key}")

        # Test 3: Download file
        print(f"\n[DOWNLOAD] Test 3: Downloading test file...")
        response = s3.get_object(Bucket=bucket_name, Key=test_key)
        downloaded_content = response['Body'].read()

        if downloaded_content == test_content:
            print("[OK] Download successful and content matches!")
        else:
            print("[ERROR] Downloaded content does not match uploaded content")
            return False

        # Test 4: List objects
        print(f"\n[INFO] Test 4: Listing objects in 'test/' prefix...")
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='test/', MaxKeys=10)

        if 'Contents' in response:
            print(f"[OK] Found {len(response['Contents'])} object(s):")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("  (No objects found)")

        # Test 5: Get object metadata
        print(f"\n[CHECK] Test 5: Getting object metadata...")
        response = s3.head_object(Bucket=bucket_name, Key=test_key)
        print(f"[OK] Object metadata:")
        print(f"  - Size: {response['ContentLength']} bytes")
        print(f"  - Content-Type: {response.get('ContentType', 'N/A')}")
        print(f"  - Last Modified: {response.get('LastModified', 'N/A')}")

        # Test 6: Delete test file (cleanup)
        print(f"\n[DELETE]  Test 6: Cleaning up test file...")
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        print("[OK] Test file deleted successfully")

        # Success!
        print("\n" + "=" * 60)
        print("[OK] All tests passed! R2 connection is working properly.")
        print("=" * 60)
        print("\n[INFO] Next steps:")
        print("  1. Use these credentials in Railway MLflow service")
        print("  2. Set MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://{bucket_name}/mlflow-artifacts")
        print("  3. Deploy MLflow service to Railway")
        print(f"\n[LINK] Artifact root: s3://{bucket_name}/mlflow-artifacts")

        return True

    except NoCredentialsError:
        print("\n[ERROR] No credentials found!")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']

        print(f"\n[ERROR] R2 Error: {error_code}")
        print(f"   Message: {error_message}")

        if error_code == 'InvalidAccessKeyId':
            print("\n[TIP] Troubleshooting:")
            print("  - Check that AWS_ACCESS_KEY_ID is correct")
            print("  - Verify the token is not expired")
            print("  - Recreate API token in R2 Dashboard if needed")

        elif error_code == 'SignatureDoesNotMatch':
            print("\n[TIP] Troubleshooting:")
            print("  - Check that AWS_SECRET_ACCESS_KEY is correct")
            print("  - Ensure no extra spaces or newlines in the key")

        elif error_code == 'NoSuchBucket':
            print("\n[TIP] Troubleshooting:")
            print(f"  - Bucket '{bucket_name}' does not exist")
            print("  - Create the bucket in Cloudflare R2 Dashboard")
            print("  - Or set R2_BUCKET_NAME to an existing bucket")

        elif error_code == 'AccessDenied':
            print("\n[TIP] Troubleshooting:")
            print("  - Check API token permissions: 'Object Read & Write'")
            print(f"  - Verify bucket scope includes '{bucket_name}'")
            print("  - Recreate token with correct permissions")

        return False

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print(f"   Type: {type(e).__name__}")
        return False


def main():
    """Main entry point."""
    success = test_r2_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
