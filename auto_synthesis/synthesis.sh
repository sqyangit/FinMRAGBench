# CUDA_VISIBLE_DEVICES="0" vllm serve Qwen/Qwen2.5-VL-7B-VRAG --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8001 --host 0.0.0.0 --limit-mm-per-prompt '{"image":20}' --served-model-name Qwen2.5-VL-7B-VRAG

# 部署检索
# CUDA_VISIBLE_DEVICES="0" python search_engine_api.py --port 8011 --industry Semiconductors
# CUDA_VISIBLE_DEVICES="0" python search_engine_api.py --port 8012 --industry Restaurants
# CUDA_VISIBLE_DEVICES="1" python search_engine_api.py --port 8013 --industry Pharmaceuticals
# CUDA_VISIBLE_DEVICES="1" python search_engine_api.py --port 8014 --industry AerospaceDefense
# CUDA_VISIBLE_DEVICES="2" python search_engine_api.py --port 8015 --industry Automobile
# CUDA_VISIBLE_DEVICES="2" python search_engine_api.py --port 8016 --industry Apparel
# CUDA_VISIBLE_DEVICES="3" python search_engine_api.py --port 8017 --industry ElectricUtilities
# CUDA_VISIBLE_DEVICES="3" python search_engine_api.py --port 8018 --industry Retail
# CUDA_VISIBLE_DEVICES="4" python search_engine_api.py --port 8019 --industry Energy
# CUDA_VISIBLE_DEVICES="4" python search_engine_api.py --port 8020 --industry Bank

# 部署rechecker小模型     tmux at -d -t gen       tmux new -s gen
# CUDA_VISIBLE_DEVICES="0" vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8001 --host 0.0.0.0 --limit-mm-per-prompt '{"image":20}' --served-model-name Qwen/Qwen3-VL-8B-Instruct
# CUDA_VISIBLE_DEVICES="1" vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8002 --host 0.0.0.0 --limit-mm-per-prompt '{"image":20}' --served-model-name Qwen/Qwen3-VL-8B-Instruct
# CUDA_VISIBLE_DEVICES="2" vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8003 --host 0.0.0.0 --limit-mm-per-prompt '{"image":20}' --served-model-name Qwen/Qwen3-VL-8B-Instruct
# CUDA_VISIBLE_DEVICES="3" vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8004 --host 0.0.0.0 --limit-mm-per-prompt '{"image":20}' --served-model-name Qwen/Qwen3-VL-8B-Instruct
# CUDA_VISIBLE_DEVICES="4" vllm serve Qwen/Qwen3-VL-8B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8005 --host 0.0.0.0 --limit-mm-per-prompt '{"image":20}' --served-model-name Qwen/Qwen3-VL-8B-Instruct

# 部署sft模型
CUDA_VISIBLE_DEVICES="0" vllm serve ${FINMRAG_SFT_MODEL:-models/finmragagent} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8001 --host 0.0.0.0 --limit-mm-per-prompt '{"image":30}' --served-model-name Qwen/Qwen3-VL-8B-SFT-complex
CUDA_VISIBLE_DEVICES="1" vllm serve ${FINMRAG_SFT_MODEL:-models/finmragagent} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8002 --host 0.0.0.0 --limit-mm-per-prompt '{"image":30}' --served-model-name Qwen/Qwen3-VL-8B-SFT-complex
CUDA_VISIBLE_DEVICES="2" vllm serve ${FINMRAG_SFT_MODEL:-models/finmragagent} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8003 --host 0.0.0.0 --limit-mm-per-prompt '{"image":30}' --served-model-name Qwen/Qwen3-VL-8B-SFT-complex
CUDA_VISIBLE_DEVICES="3" vllm serve ${FINMRAG_SFT_MODEL:-models/finmragagent} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8004 --host 0.0.0.0 --limit-mm-per-prompt '{"image":30}' --served-model-name Qwen/Qwen3-VL-8B-SFT-complex
CUDA_VISIBLE_DEVICES="4" vllm serve ${FINMRAG_SFT_MODEL:-models/finmragagent} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-batched-tokens 8192 --port 8005 --host 0.0.0.0 --limit-mm-per-prompt '{"image":30}' --served-model-name Qwen/Qwen3-VL-8B-SFT-complex

# 数据合成的前台运行示例
# CUDA_VISIBLE_DEVICES="0" python synthesis.py --industry "Semiconductors" --retriever_port 8011 --small_model_port 8001
# CUDA_VISIBLE_DEVICES="1" python synthesis.py --industry "Restaurants" --retriever_port 8012 --small_model_port 8002
# CUDA_VISIBLE_DEVICES="2" python synthesis.py --industry "Pharmaceuticals" --retriever_port 8013 --small_model_port 8003
# CUDA_VISIBLE_DEVICES="3" python synthesis.py --industry "AerospaceDefense" --retriever_port 8014 --small_model_port 8004
# CUDA_VISIBLE_DEVICES="4" python synthesis.py --industry "Automobile" --retriever_port 8015 --small_model_port 8005

# CUDA_VISIBLE_DEVICES="0" python synthesis.py --industry "Apparel" --retriever_port 8016 --small_model_port 8001
# CUDA_VISIBLE_DEVICES="1" python synthesis.py --industry "ElectricUtilities" --retriever_port 8017 --small_model_port 8002
# CUDA_VISIBLE_DEVICES="2" python synthesis.py --industry "Retail" --retriever_port 8018 --small_model_port 8003
# CUDA_VISIBLE_DEVICES="3" python synthesis.py --industry "Energy" --retriever_port 8019 --small_model_port 8004
# CUDA_VISIBLE_DEVICES="4" python synthesis.py --industry "Bank" --retriever_port 8020 --small_model_port 8005

# 使用 nohup 运行事实验证问题合成
CUDA_VISIBLE_DEVICES="0" nohup python synthesis.py --industry "Semiconductors" --retriever_port 8011 --small_model_port 8001 > log1/log1.txt 2>&1 &
CUDA_VISIBLE_DEVICES="1" nohup python synthesis.py --industry "Restaurants" --retriever_port 8012 --small_model_port 8002 > log1/log2.txt 2>&1 &
CUDA_VISIBLE_DEVICES="2" nohup python synthesis.py --industry "Pharmaceuticals" --retriever_port 8013 --small_model_port 8003 > log1/log3.txt 2>&1 &
CUDA_VISIBLE_DEVICES="3" nohup python synthesis.py --industry "AerospaceDefense" --retriever_port 8014 --small_model_port 8004 > log1/log4.txt 2>&1 &
CUDA_VISIBLE_DEVICES="4" nohup python synthesis.py --industry "Automobile" --retriever_port 8015 --small_model_port 8005 > log1/log5.txt 2>&1 &
CUDA_VISIBLE_DEVICES="0" nohup python synthesis.py --industry "Apparel" --retriever_port 8016 --small_model_port 8001 > log1/log6.txt 2>&1 &
CUDA_VISIBLE_DEVICES="1" nohup python synthesis.py --industry "ElectricUtilities" --retriever_port 8017 --small_model_port 8002 > log1/log7.txt 2>&1 &
CUDA_VISIBLE_DEVICES="2" nohup python synthesis.py --industry "Retail" --retriever_port 8018 --small_model_port 8003 > log1/log8.txt 2>&1 &
CUDA_VISIBLE_DEVICES="3" nohup python synthesis.py --industry "Energy" --retriever_port 8019 --small_model_port 8004 > log1/log9.txt 2>&1 &
CUDA_VISIBLE_DEVICES="4" nohup python synthesis.py --industry "Bank" --retriever_port 8020 --small_model_port 8005 > log1/log10.txt 2>&1 &

# 使用 nohup 运行知识推理问题合成
CUDA_VISIBLE_DEVICES="0" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Semiconductors" --retriever_port 8011 --small_model_port 8001 > log2/log1.txt 2>&1 &
CUDA_VISIBLE_DEVICES="1" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Restaurants" --retriever_port 8012 --small_model_port 8002 > log2/log2.txt 2>&1 &
CUDA_VISIBLE_DEVICES="2" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Pharmaceuticals" --retriever_port 8013 --small_model_port 8003 > log2/log3.txt 2>&1 &
CUDA_VISIBLE_DEVICES="3" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "AerospaceDefense" --retriever_port 8014 --small_model_port 8004 > log2/log4.txt 2>&1 &
CUDA_VISIBLE_DEVICES="4" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Automobile" --retriever_port 8015 --small_model_port 8005 > log2/log5.txt 2>&1 &
CUDA_VISIBLE_DEVICES="0" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Apparel" --retriever_port 8016 --small_model_port 8001 > log2/log6.txt 2>&1 &
CUDA_VISIBLE_DEVICES="1" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "ElectricUtilities" --retriever_port 8017 --small_model_port 8002 > log2/log7.txt 2>&1 &
CUDA_VISIBLE_DEVICES="2" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Retail" --retriever_port 8018 --small_model_port 8003 > log2/log8.txt 2>&1 &
CUDA_VISIBLE_DEVICES="3" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Energy" --retriever_port 8019 --small_model_port 8004 > log2/log9.txt 2>&1 &
CUDA_VISIBLE_DEVICES="4" nohup python synthesis.py --task_type "knowledge_reasoning" --industry "Bank" --retriever_port 8020 --small_model_port 8005 > log2/log10.txt 2>&1 &

# 完整版 CUDA_VISIBLE_DEVICES="4" python synthesis.py --industry "Bank" --retriever_port 8020 --small_model "Qwen/Qwen3-VL-8B-Instruct" --small_model_ip "127.0.0.1" --small_model_port 8005
python synthesis.py \
  --industry "Restaurants" \
  --retriever_port 8012 \
  --small_model "Qwen/Qwen3-VL-8B-Instruct" \
  --small_model_ip "127.0.0.1" \
  --small_model_port 8001
