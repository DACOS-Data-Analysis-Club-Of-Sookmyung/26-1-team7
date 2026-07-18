# 🏠HomeHUG — 전세사기 피해 지원 AI 법률상담 챗봇

> DACOS 26-1 심화프로젝트 — Team USIM
> 

---

## 프로젝트 개요

- **문제의식**: 전세사기 피해자는 본인이 처한 상황이 법적으로 어떤 의미를 갖는지, 어떤 절차를 밟아야 하는지 판단하기 어렵습니다. 법률 용어와 실제 피해 상황(구어체 표현) 사이의 간극이 큽니다.
- **목표**: (전세사기 "예방"이 아닌) **피해자 구제·지원**에 초점을 맞춘 상담 챗봇 구축. 사용자의 상황 설명을 법률 의도(intent)로 매핑하고, 관련 판례·가이드라인을 근거로 답변을 생성합니다.


## 기술 스택

| 레이어 | 기술 |
| --- | --- |
| Language | Python |
| Orchestration | LangGraph |
| Backend | FastAPI |
| Vector DB | Qdrant (Cloud) |
| Embedding / LLM | Gemini API |
| State / Checkpoint | SqliteSaver |
| Frontend / UI | Streamlit |
| Evaluation | Ragas |


## 주요 파이프라인 및 아키텍처
<img width="1500" height="900" alt="jeonse_chatbot_langgraph_pipeline_horizontal" src="https://github.com/user-attachments/assets/177d209d-c04a-4992-bbb6-ddf0eace61d9" />
<img width="1500" height="1000" alt="jeonse_chatbot_system_architecture" src="https://github.com/user-attachments/assets/bc914644-bc97-471b-92f0-98dad002da54" />



## 사용 데이터

| 구분 | 내용 | 데이터셋 |
| --- | --- | --- |
| **유불리 데이터** | 계약 조항·상황별로 임차인에게 유리/불리한지를 판단한 데이터 | AI-Hub 법률/규정(판결서, 약관 등) 데이터셋 |
| **특별법 기반 가이드라인** | 전세사기피해자 지원 특별법 등 관련 특별법에 근거한 대응 가이드라인 | 국토교통부 '전세사기피해자등 안내사항' |
| **국세기본법·국세징수법 판결문** | 조세채권과 임차보증금 간 우선순위 관련 판결문 | AI-Hub 법률 지식 기반 관계 데이터 데이터셋 |
| **형사 판례 데이터** | 전세사기 관련 사기죄 등 형사 판례 | Hugging Face `korean_law_open_data_precedents` 데이터셋 |

---

## 실행 방법

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 환경 변수 설정

```
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. 실행

```bash
uvicorn src.main:app --reload    # backend 디렉터리 내에서 명령어 실행
streamlit run app.py             # frontend 디렉터리 내에서 명령어 실행
```

