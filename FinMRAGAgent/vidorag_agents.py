
import os
from pathlib import Path

# Resolve paths relative to the repository root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
from PIL import Image
import json
import sys

from agent.agent_prompt import seeker_prompt,inspector_prompt_numerical, answer_prompt_numerical, inspector_prompt_table, answer_prompt_table, inspector_prompt_chart, answer_prompt_chart, inspector_prompt_knowledge, answer_prompt_knowledge, inspector_prompt_fact, answer_prompt_fact
from agent.map_dict import arrangement_map_dict,page_map_dict_normal,page_map_dict

from utils.parse_tool import extract_json
from utils.image_preprosser import concat_images_with_bbox

from json_repair import repair_json



class Seeker:
    def __init__(self, vlm):
        self.vlm = vlm
        self.seeker_multi_image = False

        if self.seeker_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict
    def run(self, query=None, images_path=None, feedback=None):

        if query is not None and images_path is not None:
            self.buffer_images = images_path
            self.query = query
            prompt = seeker_prompt.replace('{question}', self.query).replace('{page_map}', self.page_map[len(self.buffer_images)])
        
        elif feedback is not None:
            additional_information = self.query + '\n\n## Additional Information\n' + feedback
            prompt = seeker_prompt.replace('{question}', additional_information).replace('{page_map}', self.page_map[len(self.buffer_images)])        

        if self.seeker_multi_image:
            input_images = self.buffer_images
        else:
            input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]

        # import pdb;pdb.set_trace()
        times = 0
        while True:
            if times > 2:
                # return None, None, None
                raise Exception('seeker time out')
            times += 1
            # import pdb;pdb.set_trace()
            select_response = self.vlm.generate(query=prompt, image=input_images)
            print(select_response)
            select_response = repair_json(select_response, ensure_ascii=False)
            print("repair_json",select_response)
            try:
                select_response_json = extract_json(select_response)
                reason = select_response_json.get('reason', None)
                summary = select_response_json.get('summary', None)
                select_page_num = select_response_json.get('choice', None)
                if (reason is None or summary is None or not isinstance(select_page_num, list)
                        or any(not isinstance(page, int) or not 0 <= page < len(self.buffer_images)
                               for page in select_page_num)):
                    raise Exception(f'select json format error: length: {len(self.buffer_images)}')

                selected_images = [self.buffer_images[page] for page in select_page_num]
                self.buffer_images = [image for image in self.buffer_images if image not in selected_images]

            except Exception as e:
                print(e)
                print(select_response)
                print('seeker error')
                continue
            break
        
        return selected_images, summary, reason

class Inspector:
    def __init__(self, vlm):
        self.vlm = vlm
        self.inspector_multi_image = False

        if self.inspector_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

        self.buffer_images = []

    def run(self, query, images_path, inspector_prompt):
        # answer or not, (images, candidate)/feedback
        if len(self.buffer_images) == 0 and len(images_path) == 0:
            return None, None, None
        elif len(images_path) == 0:
            return 'synthesizer', None, self.buffer_images
        elif len(images_path) != 0:
            self.buffer_images.extend(images_path)

        if self.inspector_multi_image:
            input_images = self.buffer_images
        else:
            input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]

        prompt = inspector_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(self.buffer_images)])

        times = 0
        while True:
            if times >2:
                raise Exception('Inspector time out')
                # return None, None, None
            times +=1
            # import pdb;pdb.set_trace()
            response = self.vlm.generate(query=prompt,image=input_images)
            print(response)
            response = repair_json(response, ensure_ascii=False)
            print("repair_json",response)
            try:
                response_json = extract_json(response)
                # thought
                reason = response_json.get('reason',None)
                # if feedback
                info = response_json.get('information',None)
                choice = response_json.get('choice',None)
                # can answer
                answer = response_json.get('answer',None)
                ref = response_json.get('reference',None)

                if reason is None:
                    raise Exception('answer no reason')
                elif answer is not None and ref is not None:
                    if (not isinstance(ref, list) or not ref
                            or any(not isinstance(page, int) or not 0 <= page < len(self.buffer_images)
                                   for page in ref)):
                        raise Exception('ref error')
                    if len(ref) == len(self.buffer_images):
                        return 'answer', answer, self.buffer_images
                    else:
                        ref_images = [self.buffer_images[page] for page in ref]
                        return 'synthesizer', answer, ref_images
                elif info is not None and choice is not None:
                    if (not isinstance(choice, list)
                            or any(not isinstance(page, int) or not 0 <= page < len(self.buffer_images)
                                   for page in choice)):
                        raise Exception('choice error')
                    self.buffer_images = [self.buffer_images[page] for page in choice]
                    return 'seeker', info, self.buffer_images
            
            except Exception as e:
                print(e)
                print("inspector error")
                continue

class Synthesizer:
    def __init__(self, vlm):
        self.vlm = vlm
        self.synthesizer_multi_image = False
        if self.synthesizer_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

    def run(self, query, candidate_answer, ref_images, answer_prompt):
        if candidate_answer is not None:
            query = query + '\n\n## Related Information\n' + candidate_answer
        prompt = answer_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(ref_images)])

        if self.synthesizer_multi_image:
            input_images = ref_images
        else:
            input_images = [concat_images_with_bbox(ref_images, arrangement=arrangement_map_dict[len(ref_images)], scale=1, line_width=40)]
        
        for _attempt in range(3):
            final_answer_response = self.vlm.generate(query=prompt,image=input_images)
            print('final_answer_response:', final_answer_response)
            try:
                final_answer_response_json = extract_json(final_answer_response)
                reason = final_answer_response_json.get('reason',None)
                answer = final_answer_response_json.get('answer',None)
                if reason is None or answer is None :
                    raise Exception('Synthesizer time out')
                return reason, answer
            except Exception as e:
                print(e)
                print(final_answer_response)
                print("answer error")
                continue

        raise RuntimeError("Synthesizer returned invalid JSON after 3 attempts")

class ViDoRAG_Agents:
    def __init__(self, vlm, max_rounds=10):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.max_rounds = max_rounds
        self.seeker = Seeker(vlm)
        self.inspector = Inspector(vlm)
        self.synthesizer = Synthesizer(vlm)

    def run_agent(self, query, images_path, task_type):
        # import pdb;pdb.set_trace()
        # initial
        self.seeker.buffer_images = None
        self.inspector.buffer_images = []

        if task_type == "table_generation":
            inspector_prompt = inspector_prompt_table
            answer_prompt = answer_prompt_table
        elif task_type == "knowledge_reasoning":
            inspector_prompt = inspector_prompt_knowledge
            answer_prompt = answer_prompt_knowledge
        elif task_type == "numerical_reasoning":
            inspector_prompt = inspector_prompt_numerical
            answer_prompt = answer_prompt_numerical
        elif task_type == "chart_generation":
            inspector_prompt = inspector_prompt_chart
            answer_prompt = answer_prompt_chart
        elif task_type == "fact_verify":
            inspector_prompt = inspector_prompt_fact
            answer_prompt = answer_prompt_fact
        else :
            raise ValueError(f"Unsupported task type: {task_type}")

        selected_images, summary, reason = self.seeker.run(query=query, images_path=images_path)
        # iter
        for _round in range(self.max_rounds):
            status, information, images = self.inspector.run(query, selected_images, inspector_prompt)
            if status == 'answer':
                return information
            elif status == 'synthesizer':
                reason, answer = self.synthesizer.run(query, information, images, answer_prompt)
                return answer
            elif status == 'seeker':
                selected_images, summary, reason = self.seeker.run(feedback=information)
                continue
            else:
                raise ValueError('ViDoRAG found no related information')

        raise RuntimeError(f"ViDoRAG exceeded {self.max_rounds} reasoning rounds")

if __name__ == '__main__':
    from llms.llm import LLM
    vlm = LLM('Qwen/Qwen2.5-VL-7B-Instruct')
    agent = ViDoRAG_Agents(vlm)
    # re=agent.run_agent(query="报告期内，公司主营业务毛利率较上期下降多少？", images_path=['<PROJECT_PATH>','<PROJECT_PATH>','<PROJECT_PATH>'])
    re=agent.run_agent(query="2022年华兴会计师事务所对贵州茅台有限公司出具了标准无保留意见的审计报告。请结合其财报是否存在重大风险或异常事项判断该审计意见是否合理？", images_path=[os.path.join(PROJECT_ROOT, 'examples/贵州茅台2023年年度报告_51.jpg'),os.path.join(PROJECT_ROOT, 'examples/贵州茅台2023年年度报告_55.jpg')])
    print(re)
