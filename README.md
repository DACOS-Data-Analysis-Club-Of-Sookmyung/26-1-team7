# 🏠HomeHUG — 전세사기 피해 지원 AI 법률상담 챗봇

> DACOS 26-1 심화프로젝트 — Team USIM
> 

---

## 프로젝트 개요

- **문제의식**: 전세사기 피해자는 본인이 처한 상황이 법적으로 어떤 의미를 갖는지, 어떤 절차를 밟아야 하는지 판단하기 어렵습니다. 법률 용어와 실제 피해 상황(구어체 표현) 사이의 간극이 큽니다.
- **목표**: (전세사기 "예방"이 아닌) **피해자 구제·지원**에 초점을 맞춘 상담 챗봇 구축. 사용자의 상황 설명을 법률 의도(intent)로 매핑하고, 관련 판례·가이드라인을 근거로 답변을 생성합니다.

## 주요 기능

- **상황 기반 법률 상담**

사용자가 겪은 상황을 구어체 그대로 입력하면, LangGraph 기반 에이전트가 이를 법률적 의도(intent)로 분류하고 관련 절차·법령을 매핑합니다. 법률 용어를 몰라도 "집주인이 보증금을 안 돌려줘요" 같은 표현만으로 상담이 가능합니다.

- **이어서 질문하기 (멀티턴 대화)**

`SqliteSaver` 기반 세션 메모리로 이전 대화 맥락을 유지합니다. 한 번의 질문으로 끝나지 않고, 사용자가 상황을 추가로 설명하거나 "그럼 그 다음엔 어떻게 해야 하나요?"처럼 이어서 물어봐도 앞선 대화 맥락을 참고해 답변합니다.

- **관련 문서/근거 제시**

답변마다 근거가 된 판례·가이드라인·법 조항을 함께 제공합니다. 챗봇이 지어낸 답이 아니라 실제 문서에 기반했는지 사용자가 직접 확인할 수 있도록 출처를 투명하게 노출합니다.

- **자주 묻는 질문(FAQ) 선택형 진입**

"어떻게 질문해야 할지 모르겠다"는 진입장벽을 낮추기 위해, 자주 발생하는 전세사기 피해 유형(보증금 미반환, 경매·공매, 우선변제권 등)을 FAQ 형태로 제시하고 클릭만으로 상담을 시작할 수 있습니다.

- **후속 질문 추천**

답변 생성 시 후속질문을 함께 생성해, 사용자가 다음에 무엇을 물어보면 좋을지 추천합니다. 법률 지식이 없는 사용자도 자연스럽게 상담을 이어갈 수 있도록 유도합니다.

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

## 평가 결과

### RAG 성능 평가

| 평가 메트릭 | 전체 평균 |
| --- | --- |
| faithfulness | 0.5322 |
| response_relevancy | 0.8269 |
| context_precision | 0.9444 |
| context_recall | 0.9166 |

### Agent & Tool 성능 평가

| 평가 메트릭 | 전체 평균 |
| --- | --- |
| Context Utilization | 0.7061 |
| Faithfulness | 0.6028 |
| Answer Relevancy | 0.8584 |
| Factual Correctness (F1) | 0.5392 |

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


---
## 실행 화면

<img width="1905" height="912" alt="image" src="https://github.com/user-attachments/assets/7d606cec-e7f1-4cd6-a398-8c1a6afb48e2" />
<img width="1902" height="915" alt="image (1)" src="https://github.com/user-attachments/assets/37a59038-ccb4-42ef-afbc-ab3e62af52dd" />

