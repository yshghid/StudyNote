# StudyNote

Jupyter 노트북 뷰어 및 실행 환경

## 로컬 실행

### 1. 백엔드 서버 실행

```bash
cd backend
pip install -r requirements.txt
python main.py
```

백엔드: http://localhost:8000

### 2. 프론트엔드 서버 실행

```bash
cd frontend
npm install
npm run dev
```

프론트엔드: http://localhost:5173

---

## Docker 실행

```bash
docker-compose up --build
```

http://localhost:80 에서 접속

---

## AWS ECS 배포

### 사전 요구사항
- AWS CLI 설정 완료
- Terraform 설치
- GitHub Secrets 설정:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`

### 1. 인프라 생성 (Terraform)

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

### 2. 배포

`main` 브랜치에 push하면 GitHub Actions가 자동으로:
1. 테스트 & 린트 실행
2. Docker 이미지 빌드 → ECR 푸시
3. ECS 서비스 업데이트

---

## 구조

```
StudyNote/
├── backend/           # FastAPI 서버
├── frontend/          # React + Vite
├── docs/              # 노트북 및 마크다운 문서
├── data/              # 노트북 실행용 데이터
├── terraform/         # AWS 인프라 (IaC)
├── .aws/              # ECS Task Definitions
└── .github/workflows/ # CI/CD 파이프라인
```
