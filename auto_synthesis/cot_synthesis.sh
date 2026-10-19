# python cot_synthesis.py \
#   --dataset ../FinMRAGAgent/data/FinanceReport \
#   --query_file questions_rationale_labeled_merged_final_part1.json \
#   --output_file_name cot_think_search_python_data_part1.jsonl \
#   --experiment_type cot \
#   --workers_num 1 \
#   --topk 10 \
#   --small_model_ip 127.0.0.1 \
#   --small_model_port 8003 \
  
# CoT 合成的单行命令示例
# python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Semiconductors.json --output_file_name cot_think_search_python_data_part1.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Semiconductors" --task_type "table_generation" --retriever_port 8011 --small_model_port 8001
# python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Restaurants.json --output_file_name cot_think_search_python_data_part1.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Restaurants" --task_type "table_generation" --retriever_port 8012 --small_model_port 8002


# CUDA_VISIBLE_DEVICES="0" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Semiconductors.json --output_file_name cot_think_search_python_Semiconductors.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Semiconductors" --task_type "chart_generation" --retriever_port 8011 --small_model_port 8001
# CUDA_VISIBLE_DEVICES="1" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Restaurants.json --output_file_name cot_think_search_python_Restaurants.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Restaurants" --task_type "chart_generation" --retriever_port 8012 --small_model_port 8002


# CUDA_VISIBLE_DEVICES="0" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Semiconductors.json --output_file_name cot_think_search_python_Semiconductors.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Semiconductors" --task_type "fact_verify" --retriever_port 8011 --small_model_port 8001
# CUDA_VISIBLE_DEVICES="1" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Restaurants.json --output_file_name cot_think_search_python_Restaurants.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Restaurants" --task_type "fact_verify" --retriever_port 8012 --small_model_port 8002


# CUDA_VISIBLE_DEVICES="0" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Semiconductors.json --output_file_name cot_think_search_python_Semiconductors.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Semiconductors" --task_type "knowledge_reasoning" --retriever_port 8011 --small_model_port 8001
# CUDA_VISIBLE_DEVICES="1" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Restaurants.json --output_file_name cot_think_search_python_Restaurants.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Restaurants" --task_type "knowledge_reasoning" --retriever_port 8012 --small_model_port 8002


# CUDA_VISIBLE_DEVICES="0" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Semiconductors.json --output_file_name cot_think_search_python_Semiconductors.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Semiconductors" --task_type "knowledge_reasoning" --retriever_port 8011 --small_model_port 8001
# CUDA_VISIBLE_DEVICES="1" python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Restaurants.json --output_file_name cot_think_search_python_Restaurants.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Restaurants" --task_type "numerical_reasoning" --retriever_port 8012 --small_model_port 8002


# 使用 nohup 按行业运行 CoT 合成
TASK_TYPE="table_generation"


CUDA_VISIBLE_DEVICES="0" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Semiconductors.json --output_file_name cot_think_search_python_Semiconductors.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Semiconductors" --task_type "${TASK_TYPE}" --retriever_port 8011 --small_model_port 8001 > logs/${TASK_TYPE}/log_semiconductors.txt 2>&1 &

CUDA_VISIBLE_DEVICES="1" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Restaurants.json --output_file_name cot_think_search_python_Restaurants.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Restaurants" --task_type "${TASK_TYPE}" --retriever_port 8012 --small_model_port 8002 > logs/${TASK_TYPE}/log_restaurants.txt 2>&1 &

CUDA_VISIBLE_DEVICES="2" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Pharmaceuticals.json --output_file_name cot_think_search_python_Pharmaceuticals.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Pharmaceuticals" --task_type "${TASK_TYPE}" --retriever_port 8013 --small_model_port 8003 > logs/${TASK_TYPE}/log_pharmaceuticals.txt 2>&1 &

CUDA_VISIBLE_DEVICES="3" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_AerospaceDefense.json --output_file_name cot_think_search_python_AerospaceDefense.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "AerospaceDefense" --task_type "${TASK_TYPE}" --retriever_port 8014 --small_model_port 8004 > logs/${TASK_TYPE}/log_aerospacedefense.txt 2>&1 &

CUDA_VISIBLE_DEVICES="4" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Automobile.json --output_file_name cot_think_search_python_Automobile.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Automobile" --task_type "${TASK_TYPE}" --retriever_port 8015 --small_model_port 8005 > logs/${TASK_TYPE}/log_automobile.txt 2>&1 &

CUDA_VISIBLE_DEVICES="0" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Apparel.json --output_file_name cot_think_search_python_Apparel.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Apparel" --task_type "${TASK_TYPE}" --retriever_port 8016 --small_model_port 8001 > logs/${TASK_TYPE}/log_apparel.txt 2>&1 &

CUDA_VISIBLE_DEVICES="1" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_ElectricUtilities.json --output_file_name cot_think_search_python_ElectricUtilities.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "ElectricUtilities" --task_type "${TASK_TYPE}" --retriever_port 8017 --small_model_port 8002 > logs/${TASK_TYPE}/log_electricutilities.txt 2>&1 &

CUDA_VISIBLE_DEVICES="2" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Retail.json --output_file_name cot_think_search_python_Retail.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Retail" --task_type "${TASK_TYPE}" --retriever_port 8018 --small_model_port 8003 > logs/${TASK_TYPE}/log_retail.txt 2>&1 &

CUDA_VISIBLE_DEVICES="3" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Energy.json --output_file_name cot_think_search_python_Energy.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Energy" --task_type "${TASK_TYPE}" --retriever_port 8019 --small_model_port 8004 > logs/${TASK_TYPE}/log_energy.txt 2>&1 &

CUDA_VISIBLE_DEVICES="4" nohup python cot_synthesis.py --dataset ../FinMRAGAgent/data/FinanceReport --query_file questions_first_Bank.json --output_file_name cot_think_search_python_Bank.jsonl --experiment_type cot --workers_num 1 --topk 10 --industry "Bank" --task_type "${TASK_TYPE}" --retriever_port 8020 --small_model_port 8005 > logs/${TASK_TYPE}/log_bank.txt 2>&1 &

