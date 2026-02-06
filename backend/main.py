import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="StudyNote API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
INDEX_FILE = BASE_DIR / "_index.md"

# 노트북별 실행 히스토리 저장 (셀 인덱스 -> 코드)
execution_history: dict[str, dict[int, str]] = {}


class FileEntry(BaseModel):
    name: str
    type: str  # 'directory' | 'file' | 'page'
    path: str
    title: str | None = None  # title from _index.md frontmatter
    children: list["FileEntry"] | None = None


def parse_frontmatter_title(file_path: Path) -> str | None:
    """_index.md 파일에서 title을 추출"""
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Check for YAML frontmatter
        if lines and lines[0].strip() == "---":
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    break
                if line.startswith("title:"):
                    # Extract title value
                    title = line[6:].strip()
                    # Remove quotes if present
                    if (title.startswith('"') and title.endswith('"')) or \
                       (title.startswith("'") and title.endswith("'")):
                        title = title[1:-1]
                    return title
    except Exception:
        pass
    return None


def strip_frontmatter(content: str) -> str:
    """마크다운에서 YAML frontmatter 제거"""
    lines = content.split("\n")

    if not lines or lines[0].strip() != "---":
        return content

    # Find closing ---
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            # Return content after frontmatter
            return "\n".join(lines[i + 1:]).lstrip()

    return content


class ExecuteRequest(BaseModel):
    notebook_name: str
    code: str
    cell_index: int


class ExecuteResponse(BaseModel):
    success: bool
    output: str
    error: str | None = None


def scan_directory(directory: Path, relative_to: Path, depth: int = 0) -> list[FileEntry]:
    """디렉토리를 스캔하여 파일 구조를 반환"""
    entries: list[FileEntry] = []

    if not directory.exists():
        return entries

    for item in sorted(directory.iterdir()):
        if item.name.startswith(".") or item.name.startswith("_"):
            continue

        relative_path = str(item.relative_to(relative_to))

        if item.is_dir():
            # _index.md가 있으면 page로 처리
            index_file = item / "_index.md"
            has_index = index_file.exists()

            if has_index:
                # _index.md에서 title 추출
                title = parse_frontmatter_title(index_file)
                # page 타입: 하위 항목 없이 단독으로 표시
                entries.append(
                    FileEntry(
                        name=item.name,
                        type="page",
                        path=relative_path,
                        title=title,
                        children=None,
                    )
                )
            else:
                # 일반 디렉토리
                children = scan_directory(item, relative_to, depth + 1)
                entries.append(
                    FileEntry(
                        name=item.name,
                        type="directory",
                        path=relative_path,
                        children=children if children else [],
                    )
                )
        elif item.suffix == ".ipynb":
            entries.append(
                FileEntry(
                    name=item.name,
                    type="file",
                    path=relative_path,
                )
            )

    return entries


@app.get("/api/docs/structure", response_model=list[FileEntry])
async def get_docs_structure():
    """docs 디렉토리 구조를 반환"""
    return scan_directory(DOCS_DIR, DOCS_DIR)


@app.get("/api/index")
async def get_index_content():
    """_index.md 파일 내용을 반환"""
    if not INDEX_FILE.exists():
        return {"content": "# BiCode\n\nWelcome to BiCode!"}

    content = INDEX_FILE.read_text(encoding="utf-8")
    return {"content": strip_frontmatter(content)}


@app.get("/api/page")
async def get_page_content(path: str):
    """페이지(_index.md) 내용을 반환"""
    page_path = DOCS_DIR / path / "_index.md"

    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")

    content = page_path.read_text(encoding="utf-8")
    return {"content": strip_frontmatter(content), "basePath": path}


@app.get("/api/notebook")
async def get_notebook(path: str):
    """노트북 파일을 반환"""
    notebook_path = DOCS_DIR / path

    if not notebook_path.exists():
        raise HTTPException(status_code=404, detail="Notebook not found")

    if not notebook_path.suffix == ".ipynb":
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        content = notebook_path.read_text(encoding="utf-8")
        notebook = json.loads(content)
        return notebook
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid notebook format")


@app.post("/api/execute", response_model=ExecuteResponse)
async def execute_cell(request: ExecuteRequest):
    """코드 셀을 실행 - 이전 셀의 코드를 누적해서 실행"""
    # 해당 노트북의 데이터 디렉토리
    data_path = DATA_DIR / request.notebook_name

    # 노트북 히스토리 초기화
    if request.notebook_name not in execution_history:
        execution_history[request.notebook_name] = {}

    # 현재 셀 코드 저장
    execution_history[request.notebook_name][request.cell_index] = request.code

    # 현재 셀 이전의 모든 코드 수집 (셀 인덱스 순서대로)
    history = execution_history[request.notebook_name]
    previous_code = []
    for idx in sorted(history.keys()):
        if idx < request.cell_index:
            previous_code.append(history[idx])

    # 임시 파일에 코드 작성
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        # 데이터 디렉토리를 작업 디렉토리로 설정하는 코드 추가
        setup_code = f"""
import os
import sys

# 데이터 디렉토리 설정
DATA_DIR = r"{data_path}"
if os.path.exists(DATA_DIR):
    os.chdir(DATA_DIR)
    sys.path.insert(0, DATA_DIR)

"""
        f.write(setup_code)

        # 이전 셀 코드 추가 (출력 억제)
        if previous_code:
            f.write("import sys, io\n")
            f.write("_original_stdout = sys.stdout\n")
            f.write("_original_stderr = sys.stderr\n")
            f.write("sys.stdout = io.StringIO()\n")
            f.write("sys.stderr = io.StringIO()\n\n")

            for code in previous_code:
                f.write(code)
                f.write("\n\n")

            # 이전 셀 실행 후 stdout/stderr 복구
            f.write("sys.stdout = _original_stdout\n")
            f.write("sys.stderr = _original_stderr\n\n")

        # 현재 셀 코드 추가 (이제 출력이 나옴)
        f.write(request.code)
        temp_file = f.name

    try:
        # Python 실행
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=120,  # 120초 타임아웃 (누적 실행이므로 늘림)
            cwd=str(data_path) if data_path.exists() else str(BASE_DIR),
        )

        if result.returncode == 0:
            output = result.stdout
            if not output and result.stderr:
                # stderr에 경고만 있는 경우
                output = result.stderr
            return ExecuteResponse(success=True, output=output or "(No output)")
        else:
            return ExecuteResponse(
                success=False,
                output="",
                error=result.stderr or "Execution failed",
            )

    except subprocess.TimeoutExpired:
        return ExecuteResponse(
            success=False,
            output="",
            error="Execution timed out (120s limit)",
        )
    except Exception as e:
        return ExecuteResponse(
            success=False,
            output="",
            error=str(e),
        )
    finally:
        # 임시 파일 삭제
        try:
            os.unlink(temp_file)
        except:
            pass


@app.post("/api/reset")
async def reset_notebook(notebook_name: str):
    """노트북 실행 히스토리 초기화"""
    if notebook_name in execution_history:
        del execution_history[notebook_name]
    return {"status": "ok"}


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
