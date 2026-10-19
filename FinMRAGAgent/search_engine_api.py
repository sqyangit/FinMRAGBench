
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any, Optional, Union
from search_engine import SearchEngine
from tqdm import tqdm

dataset_dir = os.path.join(PROJECT_ROOT, 'FinMRAGAgent/data/FinanceReport')

# 创建 FastAPI 应用实例
app = FastAPI(
    title="Hierarchical Search Engine API",
    description="Provides search functionality using a pre-initialized HierarchicalSearchEngine.",
    version="1.0.0",
)

# 全局变量，用于存储 SearchEngine 实例
search_engine=None
industry_name = None  # Also defined when launched through uvicorn module:app.
# 在应用启动时初始化 SearchEngine
@app.on_event("startup")
async def startup_event():
    global search_engine
    print("Initializing SearchEngine...")
    search_engine = SearchEngine(dataset_dir, embed_model_name=os.environ.get('FINMRAG_EMBED_MODEL', 'vidore/colqwen2-v1.0'),industry=industry_name)
    # search_engine = SearchEngine(dataset_dir, embed_model_name='<PROJECT_PATH>')
    # search_engine = SearchEngine(dataset_dir, embed_model_name='<PROJECT_PATH>',industry=industry_name)
    # search_engine = SearchEngine(dataset_dir, embed_model_name='<PROJECT_PATH>',industry=industry_name)


class SearchRequest(BaseModel):
    query: str
    metadata_filters: Optional[Dict[str, Union[str, List[str]]]] = None

@app.post("/search", response_model=Dict[str, Any])
async def search(request: SearchRequest):
    results = search_engine.search(request.query, metadata_filters=request.metadata_filters)
    return results

# 定义搜索 API 端点
@app.get(
    "/search",
    summary="Perform a search query.",
    description="Executes a search using the initialized SearchEngine and returns the results.",
    response_model=Dict[str, Any]  # 定义响应模型，提高 API 文档的清晰度
)
async def search1(
    query: str = Query(...),
    # metadata_filters: Optional[str] = Query(None)
):

    results = search_engine.search(query)
    return results

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", type=str, default='None', help="")
    parser.add_argument("--port", type=int, default=8112, help="") #8111多模态，8222文本
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    if args.industry == 'None':
        industry_name = None
    else:
        industry_name = args.industry
    print(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
