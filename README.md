# StudyNote

Jupyter 노트북 뷰어 및 실행 환경

## 실행 방법

### 1. 백엔드 서버 실행

```bash
cd backend
pip install -r requirements.txt
python main.py
```

백엔드 서버가 http://localhost:8000 에서 실행됩니다.

### 2. 프론트엔드 서버 실행

```bash
cd frontend
npm install
npm run dev
```

프론트엔드가 http://localhost:5173 에서 실행됩니다.

## 구조

```
StudyNote/
├── backend/          # FastAPI 서버
├── frontend/         # React + Vite
├── docs/             # 노트북 및 마크다운 문서
└── data/             # 노트북 실행용 데이터
```
