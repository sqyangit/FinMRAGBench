import json
import re


def parse_tool_output(output):
    
    tool_pattern = r"<action>\s*(\{.*?\})\s*</action>"
    matches = re.findall(tool_pattern, output, re.DOTALL)
    parsed_tools = []
    for match in matches:
        try:
            # tool_data = json.loads("{" + match.strip() + "}")
            tool_data = json.loads(match.strip())
            parsed_tools.append(tool_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return parsed_tools

def extract_json_0(select_response):
    select_response = select_response.replace('```json', '').replace('```', '')
    start_index = select_response.find('{')
    end_index = select_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_str = select_response[start_index:end_index + 1]
        return json.loads(json_str)
    else:
        return json.loads(select_response)

def extract_json(response):
    """Extract json data from API response"""

    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response, re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        return json.loads(content)
    
    return json.loads(response.strip())
