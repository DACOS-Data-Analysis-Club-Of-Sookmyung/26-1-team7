
 
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff, PayloadSchemaType
from .client import get_qdrant_client


qdrant_client = get_qdrant_client()

VECTOR_PARAMS = VectorParams(
    size=3072,           # gemini-embedding-001 임베딩 차원
    distance=Distance.COSINE,
    # HNSW 인덱스 구성, 비슷한 문서 간의 효율적인 검색
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100, 
    )
)

# 계약조항 + 판결문/판례 통합
qdrant_client.create_collection(
    collection_name="kor_case_vec",
    vectors_config=VECTOR_PARAMS,
)

# 예방/대응 가이드라인
qdrant_client.create_collection(
    collection_name="kor_guideline_vec",
    vectors_config=VECTOR_PARAMS,
)

print("컬렉션 생성 완료!")
print(qdrant_client.get_collections())

for collection in ["kor_case_vec", "kor_guideline_vec"]:
    qdrant_client.create_payload_index(
        collection_name=collection,
        field_name="knowledge_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"{collection}: knowledge_type 인덱스 생성 완료")