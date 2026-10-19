#!/bin/bash

IP="127.0.0.1"  # 多模态生成服务地址
IP_CHECK="127.0.0.1"

Experiment_Type="vanilla_rag"   # vanilla_rag, gold_rag, react, tooluse, vidorag

# Qwen/Qwen2.5-VL-7B-SFT-rebuttal, Qwen/Qwen3-VL-8B-SFT-rebuttal, Qwen/Qwen3-VL-4B-SFT-rebuttal
# Qwen/Qwen3-VL-8B-SFT-otherbench, Qwen/Qwen3-VL-8B-Instruct-otherbench
MODEL="gpt-5.1"    # Qwen/Qwen3-VL-8B-SFT, Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-8B-Thinking, Qwen/Qwen3-VL-32B-Instruct, gpt-5.1, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, InternVL3_5-8B, InternVL3_5-38B, Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-32B-Instruct, MiniCPM-V-4_5, Llama-3.2-11B-Vision-Instruct, gpt-4o-2024-08-06

Search_IP="127.0.0.1"  # 检索服务地址，通过 --search_ip 传入推理入口
Search_Port=8111  # 8111, 8112

Emb_Model="vidore/colqwen2-v1.0" # vidore/colqwen2-v1.0, BAAI/bge-m3, vidore/colpali-v1.2, openbmb/VisRAG-Ret

LOG_DIR="../results/logs/${Experiment_Type}_${MODEL}"
mkdir -p ${LOG_DIR}

echo ">>> FinMRAGBench: numerical_reasoning.json"
export CUDA_VISIBLE_DEVICES=3
nohup python -u eval.py \
  --dataset FinanceReport \
  --query_file numerical_reasoning.json \
  --experiment_type $Experiment_Type \
  --embed_model_name $Emb_Model \
  --generate_ip $IP \
  --generate_port 8002 \
  --check_ip $IP_CHECK \
  --check_port 8002 \
  --generate_vlm $MODEL \
  --search_ip $Search_IP \
  --search_port $Search_Port \
  --has_filter true \
  > ${LOG_DIR}/log_num.txt 2>&1 &

# ##############################################
# echo ">>> Task 1: table_generation.json on GPU 0"
# export CUDA_VISIBLE_DEVICES=0
# nohup python -u eval.py \
#   --dataset FinanceReport \
#   --query_file table_generation.json \
#   --experiment_type $Experiment_Type \
#   --embed_model_name  $Emb_Model\
#   --generate_ip $IP \
#   --generate_port 8001 \
#   --check_ip $IP_CHECK \
#   --check_port 8001 \
#   --generate_vlm $MODEL \
#   --search_ip $Search_IP \
#   --search_port $Search_Port \
#   --has_filter true \
#   > ${LOG_DIR}/log_table.txt 2>&1 &

# ###############################################
# echo ">>> Task 2: chart_generation.json on GPU 1"
# export CUDA_VISIBLE_DEVICES=1
# nohup python -u eval.py \
#   --dataset FinanceReport \
#   --query_file chart_generation.json \
#   --experiment_type $Experiment_Type \
#   --embed_model_name $Emb_Model \
#   --generate_ip $IP \
#   --generate_port 8002 \
#   --check_ip $IP_CHECK \
#   --check_port 8002 \
#   --generate_vlm $MODEL \
#   --search_ip $Search_IP \
#   --search_port $Search_Port \
#   --has_filter true \
#   > ${LOG_DIR}/log_chart.txt 2>&1 &

# ###############################################
# echo ">>> Task 3: fact_verify.json on GPU 2"
# export CUDA_VISIBLE_DEVICES=2
# nohup python -u eval.py \
#   --dataset FinanceReport \
#   --query_file fact_verify.json \
#   --experiment_type $Experiment_Type \
#   --embed_model_name $Emb_Model \
#   --generate_ip $IP \
#   --generate_port 8003 \
#   --check_ip $IP_CHECK \
#   --check_port 8003 \
#   --generate_vlm $MODEL \
#   --search_ip $Search_IP \
#   --search_port $Search_Port \
#   --has_filter true \
#   > ${LOG_DIR}/log_fact.txt 2>&1 &

# ##########################################
# echo ">>> Task 4: knowledge_reasoning.json on GPU 3"
# export CUDA_VISIBLE_DEVICES=3
# nohup python -u eval.py \
#   --dataset FinanceReport \
#   --query_file knowledge_reasoning.json \
#   --experiment_type $Experiment_Type \
#   --embed_model_name $Emb_Model \
#   --generate_ip $IP \
#   --generate_port 8004 \
#   --check_ip $IP_CHECK \
#   --check_port 8004 \
#   --generate_vlm $MODEL \
#   --search_ip $Search_IP \
#   --search_port $Search_Port \
#   --has_filter true \
#   > ${LOG_DIR}/log_knowl.txt 2>&1 &

# ###############################################
# echo ">>> Task 5: numerical_reasoning.json on GPU 4"
# export CUDA_VISIBLE_DEVICES=4
# nohup python -u eval.py \
#   --dataset FinanceReport \
#   --query_file numerical_reasoning.json \
#   --experiment_type $Experiment_Type \
#   --embed_model_name $Emb_Model \
#   --generate_ip $IP \
#   --generate_port 8005 \
#   --check_ip $IP_CHECK \
#   --check_port 8005 \
#   --generate_vlm $MODEL \
#   --search_ip $Search_IP \
#   --search_port $Search_Port \
#   --has_filter true \
#   > ${LOG_DIR}/log_num.txt 2>&1 &

echo "All tasks launched with nohup!"
