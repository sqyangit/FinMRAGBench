import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode
from llama_index.core import SimpleDirectoryReader

from llms.vl_embedding import VL_Embedding

class Ingestion:
    def __init__(self, dataset_dir,input_prefix='ppocr',output_prefix='bge_ingestion',embed_model_name='BAAI/bge-m3'):
        self.dataset_dir = dataset_dir
        self.input_dir  = os.path.join(dataset_dir, input_prefix)
        self.output_dir = os.path.join(dataset_dir, output_prefix)
        self.output_file_format = 'document'
        self.chunk_size = 8192
        self.overlap_size = 0
        self.workers = 1
        self.reader = FlatReader()
        self.embed_model_name = embed_model_name
        # colqwen/colpali/visrag(openbmb)/qwen3-vl
        if 'vidore' in embed_model_name or 'openbmb' in embed_model_name or 'Qwen3-VL-Embedding' in embed_model_name: 
            if input_prefix == 'img':
                self.reader = SimpleDirectoryReader(input_dir = self.input_dir)
                self.pipeline = IngestionPipeline(
                    transformations=[
                        SimpleFileNodeParser(),
                        VL_Embedding(model=embed_model_name,mode='image')
                    ],
                    cache=None, # 禁用缓存，避免重复生成节点
                    disable_cache=True
                )
            else:
                self.pipeline = IngestionPipeline(
                                    transformations=[
                                        SimpleFileNodeParser(),
                                        SentenceSplitter(
                                            include_metadata=True, include_prev_next_rel=True,
                                            chunk_size=self.chunk_size,
                                            chunk_overlap=self.overlap_size,
                                            separator=' ',       
                                            paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'),
                                        VL_Embedding(model=embed_model_name,mode='text')
                                    ],
                                    cache=None, # 禁用缓存，避免重复生成节点
                                    disable_cache=True
                                )
        else:
            self.pipeline = IngestionPipeline(
                                transformations=[
                                    SimpleFileNodeParser(),
                                    SentenceSplitter(
                                        include_metadata=True, include_prev_next_rel=True,
                                        chunk_size=self.chunk_size,
                                        chunk_overlap=self.overlap_size,
                                        separator=' ',       
                                        paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'),
                                    HuggingFaceEmbedding(model_name=self.embed_model_name,trust_remote_code=True)
                                ],
                                cache=None, # 禁用缓存，避免重复生成节点
                                disable_cache=True
                            )

    def ingestion_example(self, input_file, output_file):
        # image
        if input_file.endswith('.jpg') or input_file.endswith('.png'):
            documents = self.reader.load_file(Path(input_file),self.reader.file_metadata,self.reader.file_extractor)
            nodes = self.pipeline.run(documents=documents,num_workers=1, show_progress=False)
        else: # txt
            documents=self.reader.load_data(Path(input_file))
            nodes = self.pipeline.run(documents=documents, show_progress=False)
        nodes_json = [node.to_dict() for node in nodes]
        with open(output_file, 'w') as json_file:
            json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)
        return True
    
    def ingestion_multi_session(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        file_to_process = []
        for file in os.listdir(self.input_dir):
            file_prefix,_ = os.path.splitext(file)
            input_file = os.path.join(self.input_dir, file)
            output_file = os.path.join(self.output_dir, file_prefix) + '.node'
            if not os.path.exists(output_file):
                file_to_process.append((input_file, output_file))
        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process):
                self.ingestion_example(input_file, output_file)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    result_type = future.result()
    


if __name__ == '__main__':
    root_path = './data'
    datasets = ['FinanceReport']
    for dataset in datasets:
        dataset_dir = os.path.join(root_path, dataset)

        # select a embedding model
        # ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='colqwen_ingestion',embed_model_name='<PROJECT_PATH>') # colqwen2
        ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='colqwen_ingestion',embed_model_name=os.environ.get('FINMRAG_EMBED_MODEL', 'vidore/colqwen2-v1.0')) # colpali
        # ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='visrag_ingestion',embed_model_name='<PROJECT_PATH>') # visrag

        # ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='qwen3vl2b_ingestion',embed_model_name='<PROJECT_PATH>') # qwen3 vl
        # run
        ingestion.ingestion_multi_session()

        # nohup python -u ingestion.py > ingestion_qwen_2B.log 2>&1 &
        # nohup python -u ingestion.py > ingestion_qwen_8B.log 2>&1 &

