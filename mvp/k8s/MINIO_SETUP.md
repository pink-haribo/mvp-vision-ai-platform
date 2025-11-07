# MinIO S3-Compatible Storage Setup

MinIO는 로컬 개발을 위한 S3 호환 객체 스토리지입니다. R2를 완벽하게 대체할 수 있습니다.

## 배포 상태

```bash
# MinIO Pod 확인
kubectl get pods -n storage

# MinIO Service 확인
kubectl get svc -n storage
```

## 접근 방법

### 1. MinIO Console (Web UI)

NodePort를 통한 직접 접근:
```
http://localhost:30901
```

Port-forward를 통한 접근:
```bash
kubectl port-forward -n storage svc/minio 9001:9001

# 브라우저에서 열기
http://localhost:9001
```

**로그인 정보:**
- Username: `minioadmin`
- Password: `minioadmin`

### 2. MinIO API (S3 호환)

NodePort를 통한 직접 접근:
```
http://localhost:30900
```

Port-forward를 통한 접근:
```bash
kubectl port-forward -n storage svc/minio 9000:9000

# S3 API endpoint
http://localhost:9000
```

### 3. 클러스터 내부 접근 (Training Jobs)

Training namespace의 Pod들은 다음 endpoint를 사용:
```
http://minio.storage.svc.cluster.local:9000
```

**자격증명:**
- Access Key: `minioadmin`
- Secret Key: `minioadmin`

## 생성된 버킷

```bash
# 버킷 목록 확인
kubectl exec -n storage deployment/minio -- ls -la /data/
```

**기본 버킷:**
1. `training-datasets` - 학습 데이터셋
2. `training-checkpoints` - 학습 체크포인트
3. `training-results` - 학습 결과 (모델, 메트릭)

## MinIO Client (mc) 설정

로컬에 MinIO Client가 있다면:

```bash
# MinIO alias 설정
mc alias set local http://localhost:30900 minioadmin minioadmin

# 버킷 목록 확인
mc ls local

# 파일 업로드
mc cp dataset.zip local/training-datasets/

# 파일 다운로드
mc cp local/training-results/model.pth ./
```

## Python에서 사용 (boto3)

```python
import boto3

s3_client = boto3.client(
    's3',
    endpoint_url='http://minio.storage.svc.cluster.local:9000',  # 클러스터 내부
    # endpoint_url='http://localhost:30900',  # 로컬에서 테스트
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    region_name='us-east-1'  # MinIO는 region을 사용하지 않지만 boto3는 필요
)

# 파일 업로드
s3_client.upload_file('model.pth', 'training-results', 'run_123/model.pth')

# 파일 다운로드
s3_client.download_file('training-datasets', 'dataset.zip', '/tmp/dataset.zip')

# 파일 목록
response = s3_client.list_objects_v2(Bucket='training-datasets')
for obj in response.get('Contents', []):
    print(obj['Key'])
```

## 데이터 영속성

**현재 설정:** `emptyDir` (Pod 재시작 시 데이터 삭제)

**영속적 저장소로 변경하려면:**

1. PersistentVolumeClaim 생성:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: storage
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

2. `minio-config.yaml`의 volume 섹션 수정:
```yaml
volumes:
- name: data
  persistentVolumeClaim:
    claimName: minio-pvc
```

3. 재배포:
```bash
kubectl apply -f mvp/k8s/minio-config.yaml
```

## Kubernetes Secret 확인

Training namespace에 설정된 MinIO 자격증명:

```bash
# Secret 내용 확인
kubectl get secret r2-credentials -n training -o yaml

# Decode된 값 확인
kubectl get secret r2-credentials -n training -o jsonpath='{.data.endpoint}' | base64 -d
kubectl get secret r2-credentials -n training -o jsonpath='{.data.access-key}' | base64 -d
```

## 트러블슈팅

### MinIO Pod가 시작되지 않음

```bash
# Pod 로그 확인
kubectl logs -n storage deployment/minio

# Pod 상태 확인
kubectl describe pod -n storage -l app=minio
```

### MinIO Console 접근 불가

```bash
# Port-forward로 직접 접근
kubectl port-forward -n storage svc/minio 9001:9001

# 브라우저: http://localhost:9001
```

### S3 API 연결 오류

```bash
# 클러스터 내부 DNS 확인
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- nslookup minio.storage.svc.cluster.local

# MinIO API 헬스체크
kubectl exec -n storage deployment/minio -- curl -f http://localhost:9000/minio/health/live
```

## Production 환경으로 전환

Production 환경에서는 R2 (또는 다른 S3 호환 스토리지)를 사용:

1. R2 자격증명 업데이트:
```bash
kubectl create secret generic r2-credentials \
  --from-literal=endpoint=https://YOUR_ACCOUNT.r2.cloudflarestorage.com \
  --from-literal=access-key=YOUR_R2_ACCESS_KEY \
  --from-literal=secret-key=YOUR_R2_SECRET_KEY \
  --namespace=training \
  --dry-run=client -o yaml | kubectl apply -f -
```

2. Training Job YAML은 변경 없이 동일하게 사용 (S3 호환 API)

## 다음 단계

MinIO 설정이 완료되었으므로:
1. ✅ 로컬에서 S3 호환 스토리지 사용 가능
2. ✅ Training Job에서 체크포인트 저장/로드 가능
3. ✅ 데이터셋 업로드/다운로드 가능

다음으로 샘플 Training Job을 생성하여 전체 플로우를 테스트할 수 있습니다.
