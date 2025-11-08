# GitHub Secrets Setup Guide

This document explains how to configure GitHub Secrets for automated training configuration schema uploads to Cloudflare R2.

## Required Secrets

Navigate to: **Repository → Settings → Secrets and variables → Actions → New repository secret**

### 1. R2_ENDPOINT_URL

**Description**: Cloudflare R2 API endpoint URL

**Format**: `https://xxxxx.r2.cloudflarestorage.com`

**How to get**:
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to **R2** → Your bucket
3. Click **Settings** → **Endpoint**
4. Copy the S3-compatible endpoint URL

**Example**:
```
https://1234567890abcdef.r2.cloudflarestorage.com
```

---

### 2. R2_ACCESS_KEY_ID

**Description**: R2 API Access Key ID

**How to get**:
1. Cloudflare Dashboard → **R2**
2. Click **Manage R2 API Tokens**
3. Click **Create API Token**
4. **Token Name**: `github-actions-schema-upload`
5. **Permissions**:
   - Object Read & Write
   - Apply to specific buckets only: `training-results`
6. Click **Create API Token**
7. Copy the **Access Key ID**

**Example**:
```
a1b2c3d4e5f6g7h8i9j0
```

**⚠️ Important**: Save this immediately - you won't be able to view it again!

---

### 3. R2_SECRET_ACCESS_KEY

**Description**: R2 API Secret Access Key

**How to get**:
- Same process as Access Key ID above
- Copy the **Secret Access Key** when creating the API token

**Example**:
```
a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
```

**⚠️ Important**: This is shown only once. Save it securely!

---

### 4. S3_BUCKET_RESULTS

**Description**: R2 bucket name for training results and schemas

**Value**: `training-results`

**How to verify**:
1. Cloudflare Dashboard → **R2**
2. Check bucket name in the list
3. Use exact name (case-sensitive)

---

## Setting Secrets in GitHub

### Via GitHub UI

1. Go to your repository on GitHub
2. Click **Settings** (repository settings, not your account)
3. In the left sidebar, expand **Secrets and variables**
4. Click **Actions**
5. Click **New repository secret**
6. Enter:
   - **Name**: `R2_ENDPOINT_URL` (must be exact)
   - **Value**: Your R2 endpoint URL
7. Click **Add secret**
8. Repeat for all 4 secrets

### Via GitHub CLI (Alternative)

```bash
# Set secrets using gh CLI
gh secret set R2_ENDPOINT_URL -b "https://xxxxx.r2.cloudflarestorage.com"
gh secret set R2_ACCESS_KEY_ID -b "your-access-key-id"
gh secret set R2_SECRET_ACCESS_KEY -b "your-secret-access-key"
gh secret set S3_BUCKET_RESULTS -b "training-results"
```

---

## Verification

### Check Secrets are Set

```bash
# List all secrets (values are hidden)
gh secret list

# Expected output:
# R2_ENDPOINT_URL              Updated 2025-11-08
# R2_ACCESS_KEY_ID             Updated 2025-11-08
# R2_SECRET_ACCESS_KEY         Updated 2025-11-08
# S3_BUCKET_RESULTS            Updated 2025-11-08
```

### Test the Workflow

#### Option 1: Manual Trigger

1. Go to **Actions** tab
2. Select **Upload Training Configuration Schemas**
3. Click **Run workflow**
4. Select branch: `main`
5. Click **Run workflow**

#### Option 2: Create a Test PR

```bash
# Create a test branch
git checkout -b test/schema-upload-workflow

# Make a small change to trigger the workflow
echo "# Test change" >> mvp/training/config_schemas.py

# Commit and push
git add mvp/training/config_schemas.py
git commit -m "test: trigger schema upload workflow"
git push origin test/schema-upload-workflow

# Create PR
gh pr create --title "Test: Schema Upload Workflow" --body "Testing automated schema upload"
```

**Expected Result**:
- GitHub Actions workflow runs automatically
- PR gets a comment with validation results
- Workflow shows ✅ success

---

## Troubleshooting

### Error: "AWS_S3_ENDPOINT_URL not set"

**Cause**: Secret name is incorrect

**Solution**:
- Verify secret name is exactly `R2_ENDPOINT_URL` (not `AWS_S3_ENDPOINT_URL`)
- Check for typos (case-sensitive)

### Error: "Access Denied"

**Cause**: R2 API token has insufficient permissions

**Solution**:
1. Go to Cloudflare Dashboard → R2 → Manage API Tokens
2. Find the token (or create new one)
3. Ensure permissions include:
   - ✅ Object Read
   - ✅ Object Write
4. Ensure it applies to `training-results` bucket

### Error: "Bucket not found"

**Cause**: Bucket name mismatch or bucket doesn't exist

**Solution**:
1. Verify bucket exists in Cloudflare R2
2. Check `S3_BUCKET_RESULTS` secret value matches exact bucket name
3. Bucket name is case-sensitive

### Workflow runs but doesn't upload

**Cause**: Wrong branch trigger

**Solution**:
- Workflow only uploads on `push` to `main` or `production` branches
- PRs only validate, don't upload
- Check workflow file: `.github/workflows/upload-schemas.yml`

---

## Security Best Practices

### 1. Least Privilege Access

Create dedicated R2 API token with minimal permissions:
- ✅ Object Read & Write only
- ✅ Scoped to `training-results` bucket only
- ❌ Do NOT use Admin API tokens
- ❌ Do NOT grant bucket-level permissions (create, delete buckets)

### 2. Token Rotation

Rotate R2 API tokens periodically:
```bash
# Every 90 days or when team members leave
1. Create new R2 API token in Cloudflare
2. Update GitHub Secrets
3. Revoke old token in Cloudflare
```

### 3. Audit Logs

Monitor R2 access:
1. Cloudflare Dashboard → R2 → Analytics
2. Check for unexpected upload patterns
3. Review API token usage

### 4. Separate Environments

Use different buckets/tokens for different environments:

```
Production:
  R2_ENDPOINT_URL → https://prod-xxxxx.r2.cloudflarestorage.com
  S3_BUCKET_RESULTS → training-results-prod

Staging:
  R2_ENDPOINT_URL → https://staging-xxxxx.r2.cloudflarestorage.com
  S3_BUCKET_RESULTS → training-results-staging
```

Use GitHub Environments feature:
- Settings → Environments → New environment
- Create `production` and `staging` environments
- Set environment-specific secrets

---

## Alternative: Local Development

For local testing without GitHub Actions:

```bash
cd mvp/training

# Set environment variables
export AWS_S3_ENDPOINT_URL="https://xxxxx.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export S3_BUCKET_RESULTS="training-results"

# Test upload
python scripts/upload_schema_to_storage.py --all --dry-run  # Validation only
python scripts/upload_schema_to_storage.py --all             # Actual upload
```

---

## References

- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [R2 API Tokens](https://developers.cloudflare.com/r2/api/s3/tokens/)

---

**Last Updated**: 2025-11-08
**Maintainer**: Platform Team
