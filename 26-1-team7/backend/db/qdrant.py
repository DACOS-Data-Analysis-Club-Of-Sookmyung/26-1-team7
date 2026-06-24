 # Qdrant 클라이언트 연결, 컬렉션 생성/삭제, 벡터 삽입/조회

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# 1. .env 파일의 환경 변수를 로드
load_dotenv()

# 2. os.environ을 통해 환경 변수 값을 가져옴
QDRANT_CLOUD_URL = os.environ.get("QDRANT_CLOUD_URL")
QDRANT_CLOUD_API_KEY = os.environ.get("QDRANT_CLOUD_API_KEY")

# 3. Qdrant 클라이언트를 초기화
qdrant_client = QdrantClient(
    url=QDRANT_CLOUD_URL,      
    api_key=QDRANT_CLOUD_API_KEY,
)

print(qdrant_client.get_collections())