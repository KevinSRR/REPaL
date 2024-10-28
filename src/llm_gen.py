import re
import os
import json
from model import GPTCompletionModel

def parse_raw_content_to_sent_list(raw_content, parse_def_prompts=False):
    """
    
    Returns:
        e.g., ['<ENT0>The Apple Company</ENT0> was a participant of <ENT1>the Worldwide Developers Conference</ENT1>.', '<ENT0>Michael Jordan</ENT0> participated in <ENT1>the Olympic Games</ENT1>.']
        
    """
    # split the raw string by the enumeration pattern
    split_list = re.split(r'\n\d+\.', raw_content)

    print(f'Original first parsed item: {split_list[0]}')
    if not '1.' in split_list[0]: 
        print("For the first parsed item, remove it since it does not contain '1.'.")
        split_list = split_list[1:]
    else:
        print("For the first parsed item, taking contents after '1.'.")
        split_list[0] = split_list[0].split('1.')[-1].strip()
        
        
    # # remove leading "\n+digit(s)+." or leading "1." for the first enumerated instance
    # split_list[0] = re.sub(r'^\n?\d+\.', '', split_list[0].lstrip())

    # remove leading or trailing spaces
    split_list = [item.strip() for item in split_list]

    # remove leading or trailing \"
    split_list = [item.strip().strip('\"') for item in split_list]

    
    if len(split_list) == 0:
        print("Warning! Got no qualified parsed result. Return an empty list instead!")
        return []
    # in case there are words after the enumerated examples
    split_list[-1] = split_list[-1].split('\n')[0]

    cleaned_split_list = []
    if not parse_def_prompts:
        for item in split_list:
            if '<ENT0>' in item and '</ENT0>' in item and '<ENT1>' in item and '</ENT1>' in item:
                cleaned_split_list.append(item)
                print('\t\tAccepted string in initial parsing: ', item)
            else:
                print('\t\tRejected string in initial parsing: ', item)
    else:
        for item in split_list:
            if '<ENT0>' in item and '<ENT1>' in item:
                cleaned_split_list.append(item)
                print('\t\tAccepted string in initial parsing: ', item)
            else:
                print('\t\tRejected string in initial parsing: ', item)

    return cleaned_split_list
        
def spans_intersect(span1, span2):
    """
    Return true if two spans intersect.
    The spans are assumed to be left inclusive and right exclusive.
    """
    start1, end1 = span1
    start2, end2 = span2
    return not (end1 < start2 or start1 > end2)


def parse_sub_sentence(sub_sentence, remove_tags=["<ENT0>", "</ENT0>"]):
    """ 
    parse a sentence string into a list of tokens
    at the same time, remove all tags specified by remove_tags
    """
    sub_sentence = sub_sentence.strip()

    for remove_tag in remove_tags:
        sub_sentence = sub_sentence.replace(remove_tag, '')
    sub_sentence = sub_sentence.strip()
    if len(sub_sentence) == 0:
        return []
    
    return sub_sentence.split()


# Define a function to parse each sentence into an example instance
def parse_sentence(sentence, relation):
    """ 
    parse one sentence string (ent mentions surrounded by "<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>") and return an example dict where the key of 'relation' has value as relation 
    """
    # initialize an empty dictionary for the current sentence
    sentence_dict = {"tokens": [], "h": None, "t": None, "relation": relation}

    sentence = sentence.strip()

    # find first span surrounded by <ENT0></ENT0>
    ent0_first_match = re.search(r'<ENT0>(.*?)</ENT0>', sentence) # only return first occurrence
    # find first span surrounded by <ENT1></ENT1>
    ent1_first_match = re.search(r'<ENT1>(.*?)</ENT1>', sentence)

    # if any no match found, return the invalid example 
    if ent0_first_match is None or ent1_first_match is None: return sentence_dict
    
    ent0_first_start, ent0_first_end = ent0_first_match.span()
    ent1_first_start, ent1_first_end = ent1_first_match.span()

    # if two spans intersected, return the invalid example
    if spans_intersect((ent0_first_start, ent0_first_end - 1), (ent1_first_start, ent1_first_end -1)): return sentence_dict

    tokens = []

    if ent0_first_start <= ent1_first_start: # ent0 is to the left of ent1
        left_sent = sentence[:ent0_first_start]
        left_tokens = parse_sub_sentence(left_sent, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        str_within_ent0 = sentence[ent0_first_start:ent0_first_end]
        ent0_tokens = parse_sub_sentence(str_within_ent0, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        middle_sent = sentence[ent0_first_end: ent1_first_start]
        middle_tokens = parse_sub_sentence(middle_sent, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        str_within_ent1 = sentence[ent1_first_start:ent1_first_end]
        ent1_tokens = parse_sub_sentence(str_within_ent1, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        right_sent = sentence[ent1_first_end:]
        right_tokens = parse_sub_sentence(right_sent, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        sentence_dict.update({
            'tokens': left_tokens + ent0_tokens + middle_tokens + ent1_tokens + right_tokens,
            'h': [" ".join(ent0_tokens), None, [list(range(len(left_tokens), len(left_tokens) + len(ent0_tokens)))]],
            't': [" ".join(ent1_tokens), None, [list(range(len(left_tokens) + len(ent0_tokens) + len(middle_tokens), len(left_tokens) + len(ent0_tokens) + len(middle_tokens) + len(ent1_tokens)))]], 
        })
    else: # ent1 is to the left of ent0
        left_sent = sentence[:ent1_first_start]
        left_tokens = parse_sub_sentence(left_sent, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        str_within_ent1 = sentence[ent1_first_start:ent1_first_end]
        ent1_tokens = parse_sub_sentence(str_within_ent1, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        middle_sent = sentence[ent1_first_end: ent0_first_start]
        middle_tokens = parse_sub_sentence(middle_sent, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        str_within_ent0 = sentence[ent0_first_start:ent0_first_end]
        ent0_tokens = parse_sub_sentence(str_within_ent0, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        right_sent = sentence[ent0_first_end:]
        right_tokens = parse_sub_sentence(right_sent, remove_tags=["<ENT0>", "</ENT0>", "<ENT1>", "</ENT1>"])

        sentence_dict.update({
            'tokens': left_tokens + ent1_tokens + middle_tokens + ent0_tokens + right_tokens,
            'h': [" ".join(ent0_tokens), None, [list(range(len(left_tokens) + len(ent1_tokens) + len(middle_tokens), len(left_tokens) + len(ent1_tokens) + len(middle_tokens) + len(ent0_tokens)))]],
            't': [" ".join(ent1_tokens), None, [list(range(len(left_tokens), len(left_tokens) + len(ent1_tokens)))]],
        })
    
    if len(sentence_dict['h'][2][0]) == 0 or len(sentence_dict['t'][2][0]) == 0:
        return {"tokens": [], "h": None, "t": None, "relation": relation}
    
    return sentence_dict


def parse_chat_response_to_examples(response, relation, num_examples=20):
    """
    Returns:
        list of example dicts (keys include 'tokens', 'h', 't', 'relation') generated based on response. The relation of examples is defined by relation parameter. 
    """
    raw_content = response.choices[0].message.content

    generated_sent_list = parse_raw_content_to_sent_list(raw_content=raw_content)
    generated_example_list = []
    
    if len(generated_sent_list) != num_examples:
        print(f"Generated sentences ({len(generated_sent_list)}) not equal to {num_examples}: ")
        for i in generated_sent_list: print(f'\t-\t{i}')
    
        if len(generated_sent_list) >= num_examples + 1:
            print(f"\tIn this case, we assume mistakenly parsed the generated response before the enumerated examples. So we take the latter {num_examples} results of the current parsed result")
            generated_sent_list = generated_sent_list[-num_examples:]
            
    for generated_sent in generated_sent_list:
        generated_example = parse_sentence(sentence=generated_sent, relation=relation)
        if generated_example["h"] is None or generated_example["t"] is None: 
            print("\nThis sentence can not yield a valid example: ", generated_sent)
            continue
        generated_example_list.append(generated_example)
    
    return generated_example_list


def parse_response_to_json(response_string):
    return json.loads(response_string)

def get_json_output_agent(unstructured_string, system_instruction='You are a helpful assistant designed to parse the user input string and output JSON. The JSON output should contain one field name called "Definitions" and the corresponding value is a list of strings with each string being a relation definition. An example relation definition is "<ENT0> is the friend of <ENT1>.".', task_id=0, llm_model_ckpt='gpt-4o-mini-2024-07-18'):
    LLM_model = GPTCompletionModel(
        GPT_model=llm_model_ckpt,
        save_filepath='./temp.jsonl',
        api_key=os.environ["OPENAI_API_KEY"],
    )
    LLM_model.update_call_attributes(max_tokens=4096, seed=1, temperature=0.0)

    
    response = LLM_model(input={'task_id': task_id, 'messages': [
        {'role': 'system', 'content': system_instruction}, 
        {'role': 'user', 'content': unstructured_string}
    ]}, response_format={"type": "json_object"},)
    
    try:
        return parse_response_to_json(response_string=response.choices[0].message.content)
    except:
        print("Error occurred during LLM parsing to json! Returning None!")
        return None
    
    
def parse_chat_response_to_examples_LLM_json_parser(response, relation, num_examples=20, llm_model_ckpt='gpt-4o-mini-2024-07-18'):
    raw_content = response.choices[0].message.content
    
    system_instruction = 'You are a helpful assistant designed to parse the user input string and output JSON. The JSON output should contain one field with its name as "Generated Examples". The value of this field is a list of strings with each string being a generated relation example or relation instance parsed from the user input. Do not include any associated analysis or clarifications of any relation example. An example relation example is "With a happy mood, <ENT0>Henry</ENT0> grabs some food at <ENT1>KFC</ENT1>.".'

    parsed_json = get_json_output_agent(unstructured_string=raw_content, system_instruction=system_instruction, llm_model_ckpt=llm_model_ckpt)
    if parsed_json is None: generated_sent_list_cand = []
    else: generated_sent_list_cand = parsed_json["Generated Examples"]
    
    generated_sent_list = []
    for sent_idx in range(len(generated_sent_list_cand)):
        if '<ENT0>' in generated_sent_list_cand[sent_idx] and '</ENT0>' in generated_sent_list_cand[sent_idx] and '<ENT1>' in generated_sent_list_cand[sent_idx] and '</ENT1>' in generated_sent_list_cand[sent_idx]:
            generated_sent_list.append(generated_sent_list_cand[sent_idx])
            print('\t\tAccepted string in initial parsing: ', generated_sent_list_cand[sent_idx])
        else:
            print('\t\tRejected string in initial parsing: ', generated_sent_list_cand[sent_idx])
    
    generated_example_list = []
    
    if len(generated_sent_list) != num_examples:
        print(f"Generated sentences ({len(generated_sent_list)}) not equal to {num_examples}: ")
        for i in generated_sent_list: print(f'\t-\t{i}')
    
        if len(generated_sent_list) >= num_examples + 1:
            print(f"\tIn this case, we assume mistakenly parsed the generated response before the enumerated examples. So we take the latter {num_examples} results of the current parsed result")
            generated_sent_list = generated_sent_list[-num_examples:]
            
    for generated_sent in generated_sent_list:
        generated_example = parse_sentence(sentence=generated_sent, relation=relation)
        if generated_example["h"] is None or generated_example["t"] is None: 
            print("\nThis sentence can not yield a valid example: ", generated_sent)
            continue
        generated_example_list.append(generated_example)
    
    return generated_example_list

    

def parse_chat_response_to_def_prompts(response, num_def_prompts):
    """ 
    Returns: 
        list of relation definition prompts (containing entity placeholders like <ENT0>, <ENT1>) generatted based on response. 
    """
    raw_content = response.choices[0].message.content
    
    generated_sent_list = parse_raw_content_to_sent_list(raw_content=raw_content, parse_def_prompts=True)
    for generated_sent_id in range(len(generated_sent_list)):
        if '\n' in generated_sent_list[generated_sent_id]: generated_sent_list[generated_sent_id] = generated_sent_list[generated_sent_id].strip().split('\n')[0]

    
    if len(generated_sent_list) != num_def_prompts:
        print(f"Generated definition prompts ({len(generated_sent_list)}) not equal to {num_def_prompts}: ")
        for i in generated_sent_list: print(i)
        
        if len(generated_sent_list) >= num_def_prompts + 1:
            print(f"\tIn this case, we assume mistakenly parsed the generated response before the enumerated def prompts. So we take the latter {num_def_prompts} results of the current parsed result")
            generated_sent_list = generated_sent_list[-num_def_prompts:]
    
    return generated_sent_list


def parse_chat_response_to_def_prompts_v1(response, num_def_prompts):
    """ 
    Returns: 
        list of relation definition prompts (containing entity placeholders like <ENT0>, <ENT1>) generatted based on response. 
    """
    raw_content = response.choices[0].message.content
    
    generated_sent_list = parse_raw_content_to_sent_list(raw_content=raw_content, parse_def_prompts=True)
    for generated_sent_id in range(len(generated_sent_list)):
        if '\n' in generated_sent_list[generated_sent_id]: generated_sent_list[generated_sent_id] = generated_sent_list[generated_sent_id].strip().split('\n')[0]

        if '\"' in generated_sent_list[generated_sent_id]: 
            if generated_sent_list[generated_sent_id].count('\"') % 2 == 1:
                generated_sent_list[generated_sent_id] = generated_sent_list[generated_sent_id].strip().split('\"')[0]
            else:
                generated_sent_list[generated_sent_id] = generated_sent_list[generated_sent_id][:generated_sent_list[generated_sent_id].rfind('\"')].strip() + '\"'
                
    
    
    if len(generated_sent_list) != num_def_prompts:
        print(f"Generated definition prompts ({len(generated_sent_list)}) not equal to {num_def_prompts}: ")
        for i in generated_sent_list: print(i)
        
        if len(generated_sent_list) >= num_def_prompts + 1:
            print(f"\tIn this case, we assume mistakenly parsed the generated response before the enumerated def prompts. So we take the latter {num_def_prompts} results of the current parsed result")
            generated_sent_list = generated_sent_list[-num_def_prompts:]
    
    return generated_sent_list



def parse_chat_response_to_def_prompts_LLM_json_parser(response, num_def_prompts, llm_model_ckpt, integrate_explanations=True):
    raw_content = response.choices[0].message.content
    
    system_instruction = 'You are a helpful assistant designed to parse the user input string and output JSON. The JSON output should contain one field called "Definitions" and the corresponding value is a list of strings with each string being a relation definition. An example relation definition is "<ENT0> is the venue of <ENT1> (a conference).". Correspondingly, the JSON should contain another field called "Supplementary Explanations" whose value is a list of strings with each string being the supplementary explanation or clarification of the corresponding relation definition (with same index in list) in the value of "Definitions" field. The element string in the value list of "Supplementary Explanations" shoud be an empty string if no explanation for relation found in input or if there are only explanations for particular relation examples. The value list of "Definitions" and the value list of "Supplementary Explanations" should have the same length.'
    
    
    parsed_json = get_json_output_agent(unstructured_string=raw_content, system_instruction=system_instruction, llm_model_ckpt=llm_model_ckpt)
    
    
    if parsed_json is None: 
        generated_sent_list_cand = []
        generated_sent_list_explanations_cand = []
    else:
        assert len(parsed_json["Definitions"]) == len(parsed_json["Supplementary Explanations"])
        generated_sent_list_cand = parsed_json["Definitions"]
        generated_sent_list_explanations_cand = parsed_json["Supplementary Explanations"]
    
    generated_sent_list, generated_sent_list_explanations = [], []
    
    for sent_idx in range(len(generated_sent_list_cand)):
        if '<ENT0>' in generated_sent_list_cand[sent_idx] and '<ENT1>' in generated_sent_list_cand[sent_idx]:
            generated_sent_list.append(generated_sent_list_cand[sent_idx])
            generated_sent_list_explanations.append(generated_sent_list_explanations_cand[sent_idx])
            print('\t\tAccepted string in initial parsing: ', generated_sent_list_cand[sent_idx])
            print('\t\t\t\t\t\twith parsed explanation: ', generated_sent_list_explanations_cand[sent_idx])
        else:
            print('\t\tRejected string in initial parsing: ', generated_sent_list_cand[sent_idx])
    
    if integrate_explanations:
        for sent_idx in range(len(generated_sent_list)):
            if len(generated_sent_list_explanations[sent_idx].strip()) > 0:
                generated_sent_list[sent_idx] = generated_sent_list[sent_idx] + f" (Additional relation definition explanation: {generated_sent_list_explanations[sent_idx]})"
            
                
                
    if len(generated_sent_list) != num_def_prompts:
        print(f"Generated definition prompts ({len(generated_sent_list)}) not equal to {num_def_prompts}: ")
        for i in generated_sent_list: print(i)
        
        if len(generated_sent_list) >= num_def_prompts + 1:
            print(f"\tIn this case, we assume mistakenly parsed the generated response before the enumerated def prompts. So we take the latter {num_def_prompts} results of the current parsed result")
            generated_sent_list = generated_sent_list[-num_def_prompts:]
    
    return generated_sent_list
