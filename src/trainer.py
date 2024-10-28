import os
from tqdm import tqdm, trange
from utils import (
    set_seed,
    compute_binary_f1,
    get_elbow_x_value,
    update_model_state_dict,
    get_so_tagged_example,
    plot_prediction_distributions,
    find_free_port,
    remove_repeats_based_on_inference_ids,
)

from model import (
    NLIBasedSimilarityModel,
    GPTCompletionModel,
    append_to_jsonl,
    load_jsonl,
)
import random
from transformers import AdamW, get_linear_schedule_with_warmup

from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from string import Template
import re
import gc         # garbage collect library

import math
import openai

from torch.utils.tensorboard import SummaryWriter



from dataloader import REDataLoader


#### for Pytorch DDP ####
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
#### for Pytorch DDP ####



#### for run_snowball_step1 ####
import spacy
import multiprocessing
from pattern_learning import (
    is_ascending,
    remove_duplicated_invalid_examples,
    remove_duplicated_invalid_examples_v1,
    get_shortest_dep_path_v3,
    linearize_ent_dep_path,
    linearize_ent_dep_path_w_mask,
    get_SBERT_embedding,
    get_KMeans_clusters,
    select_valid_distinct_examples,
)

from llm_gen import (
    parse_chat_response_to_examples,
    parse_chat_response_to_examples_LLM_json_parser,
    parse_chat_response_to_def_prompts,
    parse_chat_response_to_def_prompts_v1,
    parse_chat_response_to_def_prompts_LLM_json_parser,
)
#### for run_snowball_step1 ####



def ddp_setup(rank, world_size, master_port=12355):
    """
    Args:
        rank: Unique identifier of each process
        world_size: total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)   


class ModelTrainer:

    def __init__(self, args, model, dataloader):
        """
        Initialize the model trainer
        """
        # set the seed and ensure the deterministic behavior
        set_seed(args.seed)


        self.args = args 
        self.model = model
        self.dataloader: REDataLoader = dataloader

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def save_model(self, save_path, pytorch_save_dict):
        """
        Save the given pytorch_save_dict to the save_path
        """ 
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        print(f'Saving to {save_path}')
        torch.save(pytorch_save_dict, save_path)


    def LLM_init_pos_generation_parallel(self, rel_set, rel_def_prompt_list, save_suffix='', num_init_pos_examples=20, llm_ckpt_folder=None):
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat_parallel.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model.update_call_attributes(max_tokens=4096, seed=self.args.seed)
        
        max_num_attempts = 3
        
        gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${rel_def_prompt}\". Under sentence-level relation extraction setting, generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Do not overfit the pattern of the definition. Try as many different relation patterns or relation expressions as possible."),
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${rel_def_prompt}\". Under sentence-level relation extraction setting, generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Other content requirements:\n\n1. Do not overfit the pattern of definition. Try as many different relation patterns or relation expressions as possible.\n2. Generate rich and informative related contexts before and after each entity."),
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${rel_def_prompt}\". Under sentence-level relation extraction setting, generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Other content requirements:\n\n1. Do not overfit the pattern of definition. Try as many different relation patterns or relation expressions as possible.\n2. Generate rich and informative related contexts before and after each entity.\n3. The relation patterns or relation expressions should be implicit or complicated")
        ]
        
        num_init_pos_examples_adjusted = math.ceil(num_init_pos_examples / len(gen_template_list))
        
        
        task_id = 0
        input_list = []
        input_list_x_elt = []
        for rel, rel_def_prompt in zip(rel_set, rel_def_prompt_list):
            for gen_template_id, gen_template in enumerate(gen_template_list):
                prompt = gen_template.substitute(rel_def_prompt=rel_def_prompt, num_examples=num_init_pos_examples_adjusted)
                input_list.append({
                    'task_id': task_id,
                    'messages': [
                        {"role": "user", "content": prompt}
                    ]
                })
                input_list_x_elt.append(f"{rel_set.index(rel)}-{gen_template_id}")
                task_id += 1
                
        output_list = LLM_model.async_calls(input_list=input_list)
        current_best_output_list = deepcopy(output_list)
        rel_examples = [] # list of list of examples with each sub list corresponds to each templates

        input_list_to_regen = []
        input_list_to_regen_x_elt = []
        
        
        current_num_attempts = 0
        for input_idx, (input_x_elt, output_response) in enumerate(zip(input_list_x_elt, output_list)):
            rel_id = int(input_x_elt.split('-')[0])
            gen_template_id = int(input_x_elt.split('-')[1])
            rel = rel_set[rel_id]
            
            temp_rel_examples = parse_chat_response_to_examples(response=deepcopy(output_response), relation=rel, num_examples=deepcopy(num_init_pos_examples_adjusted))
            if len(temp_rel_examples) == num_init_pos_examples_adjusted: 
                rel_examples.append(deepcopy(temp_rel_examples))
            else:
                input_list_to_regen.append(deepcopy(input_list[input_idx]))
                input_list_to_regen_x_elt.append(deepcopy(input_x_elt))
                rel_examples.append(None)
        
        while (None in rel_examples) and current_num_attempts <= max_num_attempts:
            print(f"Current attempt: {current_num_attempts} with #invalid inputs to regenerate={rel_examples.count(None)}")
            assert rel_examples.count(None) == len(input_list_to_regen)
            
            output_list_to_regen = LLM_model.async_calls(input_list=input_list_to_regen)
            
            input_list_to_regen_temp = []
            input_list_to_regen_x_elt_temp = []
            
            for input_to_regen_idx, (input_x_elt_to_regen, output_response_to_regen) in enumerate(zip(input_list_to_regen_x_elt, output_list_to_regen)):
                rel_id = int(input_x_elt_to_regen.split('-')[0])
                gen_template_id = int(input_x_elt_to_regen.split('-')[1])
                rel = rel_set[rel_id]
                
                absolute_idx = input_list_x_elt.index(input_x_elt_to_regen)
                
                temp_rel_examples = parse_chat_response_to_examples(response=deepcopy(output_response_to_regen), relation=rel, num_examples=deepcopy(num_init_pos_examples_adjusted))
                
                if len(temp_rel_examples) == num_init_pos_examples_adjusted: 
                    assert rel_examples[absolute_idx] is None
                    rel_examples[absolute_idx] = deepcopy(temp_rel_examples)
                    current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
                else:
                    input_list_to_regen_temp.append(deepcopy(input_list_to_regen[input_to_regen_idx]))
                    input_list_to_regen_x_elt_temp.append(deepcopy(input_x_elt_to_regen))
                    
                    
                    if abs(len(temp_rel_examples) - num_init_pos_examples_adjusted) <= 2 and len(temp_rel_examples) > len(parse_chat_response_to_examples(response=deepcopy(current_best_output_list[absolute_idx]), relation=rel, num_examples=deepcopy(num_init_pos_examples_adjusted))):
                        current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
                    
            input_list_to_regen = deepcopy(input_list_to_regen_temp)
            input_list_to_regen_x_elt = deepcopy(input_list_to_regen_x_elt_temp)

            current_num_attempts += 1
        
        rel_examples_output = []
        for rel_example_idx, (input_x_elt, rel_example_list) in enumerate(zip(input_list_x_elt, rel_examples)):
            rel_id = int(input_x_elt.split('-')[0])
            gen_template_id = int(input_x_elt.split('-')[1])
            rel = rel_set[rel_id]
            
            if gen_template_id == 0: rel_examples_output.append([])
            
            if rel_example_list is not None:
                rel_examples_output[rel_id].extend(rel_example_list)
            else:
                rel_examples_output[rel_id].extend(parse_chat_response_to_examples(response=deepcopy(current_best_output_list[rel_example_idx]), relation=rel, num_examples=deepcopy(num_init_pos_examples_adjusted)))
            
            response = current_best_output_list[rel_example_idx]
            response_dump = response.model_dump()
            response_dump.update({'prompt_input': input_list[rel_example_idx], 'relation': rel, 'relative_id': input_x_elt})
            append_to_jsonl(
                data=response_dump,
                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
            )
            
        return rel_examples_output             
            

    def LLM_init_neg_def_generation_parallel(self, rel_set, rel_def_prompt_list, neg_def_gen_template_list, num_neg_rels_to_generate_adjusted, LLM_model, max_num_attempts, task_id, llm_ckpt_folder):
        
        input_list = []
        input_list_x_elt = []
        
        for rel, rel_def_prompt in zip(rel_set, rel_def_prompt_list):
            for neg_def_gen_template_id, neg_def_gen_template in enumerate(neg_def_gen_template_list):
                neg_def_gen_prompt = neg_def_gen_template.substitute(pos_rel_def_prompt=rel_def_prompt, num_neg_rels=num_neg_rels_to_generate_adjusted)
                
                input_list.append({
                    'task_id': task_id,
                    'messages': [
                        {"role": "user", "content": neg_def_gen_prompt}
                    ]
                })
                input_list_x_elt.append(f"neg_def_{rel_set.index(rel)}-{neg_def_gen_template_id}")
                task_id += 1

        output_list = LLM_model.async_calls(input_list=input_list)
        current_best_output_list = deepcopy(output_list)
        
        neg_def_prompts = []
        
        
        input_list_to_regen = []
        input_list_to_regen_x_elt = []
        
        current_num_attempts = 0
        for input_idx, (input_x_elt, output_response) in enumerate(zip(input_list_x_elt, output_list)):
            rel_id = int(input_x_elt[8:].split('-')[0])
            neg_def_gen_template_id = int(input_x_elt[8:].split('-')[1])
            rel = rel_set[rel_id]
            
            temp_neg_rel_def_prompts = parse_chat_response_to_def_prompts(response=deepcopy(output_response), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted))
            if len(temp_neg_rel_def_prompts) == num_neg_rels_to_generate_adjusted: neg_def_prompts.append(deepcopy(temp_neg_rel_def_prompts))
            else:
                input_list_to_regen.append(deepcopy(input_list[input_idx]))
                input_list_to_regen_x_elt.append(deepcopy(input_x_elt))
                neg_def_prompts.append(None)
            
        while (None in neg_def_prompts) and current_num_attempts <= max_num_attempts:
            print(f"Current attempt: {current_num_attempts} with #invalid inputs to regenerate={neg_def_prompts.count(None)}")
            assert neg_def_prompts.count(None) == len(input_list_to_regen)
            
            output_list_to_regen = LLM_model.async_calls(input_list=input_list_to_regen)
            input_list_to_regen_temp = []
            input_list_to_regen_x_elt_temp = []
            
            for input_to_regen_idx, (input_x_elt_to_regen, output_response_to_regen) in enumerate(zip(input_list_to_regen_x_elt, output_list_to_regen)):
                rel_id = int(input_x_elt_to_regen[8:].split('-')[0])
                neg_def_gen_template_id = int(input_x_elt_to_regen[8:].split('-')[1])
                rel = rel_set[rel_id]

                absolute_idx = input_list_x_elt.index(input_x_elt_to_regen)
                
                temp_neg_rel_def_prompts = parse_chat_response_to_def_prompts(response=deepcopy(output_response_to_regen), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted))
                
                if len(temp_neg_rel_def_prompts) == num_neg_rels_to_generate_adjusted: 
                    assert neg_def_prompts[absolute_idx] is None
                    neg_def_prompts[absolute_idx] = deepcopy(temp_neg_rel_def_prompts)
                    current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
                else:
                    input_list_to_regen_temp.append(deepcopy(input_list_to_regen[input_to_regen_idx]))
                    input_list_to_regen_x_elt_temp.append(deepcopy(input_x_elt_to_regen))
                    
                    if abs(len(temp_neg_rel_def_prompts) - num_neg_rels_to_generate_adjusted) <= 2 and len(temp_neg_rel_def_prompts) > len(parse_chat_response_to_def_prompts(response=deepcopy(current_best_output_list[absolute_idx]), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted))):
                        current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
            
            input_list_to_regen = deepcopy(input_list_to_regen_temp)
            input_list_to_regen_x_elt = deepcopy(input_list_to_regen_x_elt_temp)
            
            current_num_attempts += 1
        
        neg_def_prompts_output = []
        for neg_def_prompts_idx, (input_x_elt, neg_def_prompt_list) in enumerate(zip(input_list_x_elt, neg_def_prompts)):
            rel_id = int(input_x_elt[8:].split('-')[0])
            neg_def_gen_template_id = int(input_x_elt[8:].split('-')[1])
            rel = rel_set[rel_id]
            
            if neg_def_gen_template_id == 0: neg_def_prompts_output.append([])
            
            if neg_def_prompt_list is not None:
                neg_def_prompts_output[rel_id].extend(neg_def_prompt_list)
            else:
                neg_def_prompts_output[rel_id].extend(parse_chat_response_to_def_prompts(response=deepcopy(current_best_output_list[neg_def_prompts_idx]), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted)))
            
            response = current_best_output_list[neg_def_prompts_idx]
            response_dump = response.model_dump()
            response_dump.update({'prompt_input': input_list[neg_def_prompts_idx], 'relation': f"{rel}->!{rel}", 'relative_id': input_x_elt})
            append_to_jsonl(
                data=response_dump,
                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat_neg_def.jsonl'),
            )
            
        return neg_def_prompts_output, task_id
    
    
    def LLM_init_neg_generation_parallel(self, rel_set, rel_def_prompt_list, save_suffix='', num_neg_rels_to_generate=3, num_init_neg_examples=20, llm_ckpt_folder=None):
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat_parallel.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        ) 
        LLM_model.update_call_attributes(max_tokens=4096, seed=self.args.seed)
        
        
        max_num_attempts = 3
        
        
        neg_def_gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\". Generate ${num_neg_rels} challenging negative binary relation definitions in the same format (numbered from 1 to ${num_neg_rels}). Other requirements:\n\n1. Generated negative relations should maintain a connection to the theme of the positive relation while introducing a meaningful and unambiguous difference.\n2. Generated negative relations should be semantically close to the positive relation.\n3. Generated negative relations should contain necessary entity type information if needed for accurateness.\n\nFirst give your brief analysis (if any) and then list your numbered generated negative relation definitions (do not mix analysis and enumerated definitions)."),
        ]
        
        gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${rel_def_prompt}\". Under sentence-level relation extraction setting, generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Other content requirements:\n\n1. Do not overfit the pattern of definition. Try as many different relation patterns or relation expressions as possible.\n2. Generate rich and informative related contexts before and after each entity."),
        ]
        
        
        num_init_neg_examples_adjusted = math.ceil(num_init_neg_examples / len(gen_template_list) / num_neg_rels_to_generate)
        num_neg_rels_to_generate_adjusted = math.ceil(num_neg_rels_to_generate / len(neg_def_gen_template_list))
        

        # first generate the negative rel def prompts
        task_id = 0
        neg_def_prompts, task_id = self.LLM_init_neg_def_generation_parallel(rel_set=rel_set, rel_def_prompt_list=rel_def_prompt_list, neg_def_gen_template_list=neg_def_gen_template_list, num_neg_rels_to_generate_adjusted=num_neg_rels_to_generate_adjusted, LLM_model=LLM_model, max_num_attempts=max_num_attempts, task_id=task_id, llm_ckpt_folder=llm_ckpt_folder)
        
        input_list = []
        input_list_x_elt = []
        
        # then generate neg rel examples
        assert len(neg_def_prompts) == len(rel_set)
        for rel_id, rel in enumerate(rel_set):
            rel_neg_def_prompts = neg_def_prompts[rel_id]
            assert len(rel_neg_def_prompts) == num_neg_rels_to_generate
            for rel_neg_def_prompt_id, rel_neg_def_prompt in enumerate(rel_neg_def_prompts):
                for gen_template_id, gen_template in enumerate(gen_template_list):
                    prompt = gen_template.substitute(rel_def_prompt=rel_neg_def_prompt, num_examples=num_init_neg_examples_adjusted)
                    input_list.append({
                        'task_id': task_id,
                        'messages': [
                            {"role": "user", "content": prompt}
                        ]
                    })
                    
                    input_list_x_elt.append(f"neg_exp_{rel_id}-{rel_neg_def_prompt_id}-{gen_template_id}")
                    task_id += 1
        
        output_list = LLM_model.async_calls(input_list=input_list)
        current_best_output_list = deepcopy(output_list)
        
        
        rel_examples = [] # list of list of examples with each sub list corresponds to each generation template 
        
        input_list_to_regen = []
        input_list_to_regen_x_elt = []
        
        current_num_attempts = 0
        for input_idx, (input_x_elt, output_response) in enumerate(zip(input_list_x_elt, output_list)):
            rel_id = int(input_x_elt[8:].split('-')[0])
            rel_neg_def_prompt_id = int(input_x_elt[8:].split('-')[1])
            gen_template_id = int(input_x_elt[8:].split('-')[2])
            rel = rel_set[rel_id]
            
            
            temp_rel_examples = parse_chat_response_to_examples(response=deepcopy(output_response), relation=f"!{rel}", num_examples=num_init_neg_examples_adjusted)
            if len(temp_rel_examples) == num_init_neg_examples_adjusted:
                rel_examples.append(deepcopy(temp_rel_examples))
            else:
                input_list_to_regen.append(deepcopy(input_list[input_idx]))
                input_list_to_regen_x_elt.append(deepcopy(input_x_elt))
                rel_examples.append(None)
                

        while (None in rel_examples) and current_num_attempts <= max_num_attempts:
            print(f"Current attempt: {current_num_attempts} with #invalid inputs to regenerate={rel_examples.count(None)}")
            assert rel_examples.count(None) == len(input_list_to_regen)
            
            output_list_to_regen = LLM_model.async_calls(input_list=input_list_to_regen)
            
            input_list_to_regen_temp = []
            input_list_to_regen_x_elt_temp = []
            
            for input_to_regen_idx, (input_x_elt_to_regen, output_response_to_regen) in enumerate(zip(input_list_to_regen_x_elt, output_list_to_regen)):
                rel_id = int(input_x_elt_to_regen[8:].split('-')[0])
                rel_neg_def_prompt_id = int(input_x_elt_to_regen[8:].split('-')[1])
                gen_template_id = int(input_x_elt_to_regen[8:].split('-')[2])
                rel = rel_set[rel_id]
                
                absolute_idx = input_list_x_elt.index(input_x_elt_to_regen)

                temp_rel_examples = parse_chat_response_to_examples(response=deepcopy(output_response_to_regen), relation=f"!{rel}", num_examples=deepcopy(num_init_neg_examples_adjusted))
                
                if len(temp_rel_examples) == num_init_neg_examples_adjusted:
                    assert rel_examples[absolute_idx] is None
                    rel_examples[absolute_idx] = deepcopy(temp_rel_examples)
                    current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
                else:
                    input_list_to_regen_temp.append(deepcopy(input_list_to_regen[input_to_regen_idx]))
                    input_list_to_regen_x_elt_temp.append(deepcopy(input_x_elt_to_regen))
                    
                    if abs(len(temp_rel_examples) - num_init_neg_examples_adjusted) <= 2 and len(temp_rel_examples) > len(parse_chat_response_to_examples(response=deepcopy(current_best_output_list[absolute_idx]), relation=f"!{rel}", num_examples=deepcopy(num_init_neg_examples_adjusted))):
                        current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
            
            input_list_to_regen = deepcopy(input_list_to_regen_temp)
            input_list_to_regen_x_elt = deepcopy(input_list_to_regen_x_elt_temp)
            
            current_num_attempts += 1
            
        rel_examples_output = []
        for rel_example_idx, (input_x_elt, rel_example_list) in enumerate(zip(input_list_x_elt, rel_examples)):
            rel_id = int(input_x_elt[8:].split('-')[0])
            rel_neg_def_prompt_id = int(input_x_elt[8:].split('-')[1])
            gen_template_id = int(input_x_elt[8:].split('-')[2])
            rel = rel_set[rel_id]
            
            if rel_neg_def_prompt_id == 0: rel_examples_output.append([])
            
            if rel_example_list is not None:
                rel_examples_output[rel_id].extend(rel_example_list)
            else:
                rel_examples_output[rel_id].extend(parse_chat_response_to_examples(response=deepcopy(current_best_output_list[rel_example_idx]), relation=f"!{rel}", num_examples=deepcopy(num_init_neg_examples_adjusted)))
                
            
            response = current_best_output_list[rel_example_idx]
            response_dump = response.model_dump()
            response_dump.update({'prompt_input': input_list[rel_example_idx], 'relation': f"!{rel}", 'relative_id': input_x_elt})
            append_to_jsonl(
                data=response_dump,
                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat_neg_exp.jsonl'),
            )
            
        
        torch.save(neg_def_prompts, os.path.join(llm_ckpt_folder, 'neg_def_prompts.pt'))
        return rel_examples_output
    
    
    def LLM_init_neg_generation(self, rel_set, rel_def_prompt_list, save_suffix='', num_neg_rels_to_generate=3, num_init_neg_examples=20, llm_ckpt_folder=None):
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model.update_call_attributes(max_tokens=4096, seed=self.args.seed)
        
        max_num_attempts = 3

        neg_def_gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\". Generate ${num_neg_rels} challenging negative binary relation definitions in the same format (numbered from 1 to ${num_neg_rels}). Other requirements: 1. Generated negative relations should maintain a connection to the theme of the positive relation while introducing a meaningful and unambiguous difference. 2. Generated negative relations should be semantically close to the positive relation. 3. Generated negative relations should contain necessary entity type information if needed for accurateness. First give your brief analysis (if any) and then list your numbered generated negative relation definitions (do not mix analysis and enumerated definitions)."),
        ]
        
        gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${rel_def_prompt}\". Under sentence-level relation extraction setting, generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Other content requirements: 1. Do not overfit the pattern of definition. Try as many different relation patterns or relation expressions as possible. 2. Generate rich and informative related contexts before and after each entity."),
        ]
        
        num_init_neg_examples_adjusted = math.ceil(num_init_neg_examples / len(gen_template_list) / num_neg_rels_to_generate)
        num_neg_rels_to_generate_adjusted = math.ceil(num_neg_rels_to_generate / len(neg_def_gen_template_list))
        
        rel_examples = [] # list of list of examples with each sub list corresponds to each rel in rel_set
        neg_def_prompts = []
        
        
        task_id = 0
        for rel, rel_def_prompt in zip(rel_set, rel_def_prompt_list):
            rel_example_list = []
            
            # first generate neg relation definitions
            neg_rel_def_prompts = []
            
            for neg_def_gen_template_id, neg_def_gen_template in enumerate(neg_def_gen_template_list):
                neg_def_gen_prompt = neg_def_gen_template.substitute(pos_rel_def_prompt=rel_def_prompt, num_neg_rels=num_neg_rels_to_generate_adjusted)
            
                qualified_flag = False
                current_num_attempts = 0
                least_neg_rel_def_prompts = []
                least_response_dump = {}
                
                while qualified_flag is not True and current_num_attempts <= max_num_attempts:
                    try:
                        print(f"Attempt {current_num_attempts} for generating negative relation definitions for relation {rel}")
                        current_num_attempts += 1
                        
                        response = LLM_model(input={'task_id': task_id, 'messages': [{'role': 'user', 'content': deepcopy(neg_def_gen_prompt)}]})
                        
                        temp_neg_rel_def_prompts = parse_chat_response_to_def_prompts(response=response, num_def_prompts=num_neg_rels_to_generate_adjusted)
                        
                        response_dump = response.model_dump()
                        response_dump.update({'prompt_input': {'task_id': task_id, 'messages': [{'role': 'user', 'content': neg_def_gen_prompt}]}, 'relation': f"{rel}->!{rel}", 'relative_id': neg_def_gen_template_id})
                        
                        if len(temp_neg_rel_def_prompts) == num_neg_rels_to_generate_adjusted:
                            append_to_jsonl(
                                data=response_dump,
                                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                            )
                            neg_rel_def_prompts.extend(temp_neg_rel_def_prompts)
                            qualified_flag = True
                        else:
                            if abs(len(temp_neg_rel_def_prompts) - num_neg_rels_to_generate_adjusted) <= 2 and len(temp_neg_rel_def_prompts) > len(least_neg_rel_def_prompts):
                                least_neg_rel_def_prompts = temp_neg_rel_def_prompts
                                least_response_dump = response_dump
                    except openai.InternalServerError as e:
                        print("An Internal Server Error occurred when calling openai api:", e)
                
                if not qualified_flag:
                    neg_rel_def_prompts.extend(least_neg_rel_def_prompts)
                    append_to_jsonl(
                        data=least_response_dump,
                        filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                    )
                    print(f"Eventually LLM generation not qualified: Relation={rel} with #generated results={len(least_neg_rel_def_prompts)}. Use the sub-optimal generated results instead.")
                
                task_id += 1
                
            print(f'Parsed negative relation definitions for {rel}')
            for neg_rel_def_prompt_to_print in neg_rel_def_prompts: print(f"\t-\t{neg_rel_def_prompt_to_print}")
            
            neg_def_prompts.append(neg_rel_def_prompts)
            
            for neg_rel_def_prompt_id, neg_rel_def_prompt in enumerate(neg_rel_def_prompts):
                for gen_template_id, gen_template in enumerate(gen_template_list):
                    prompt = gen_template.substitute(rel_def_prompt=neg_rel_def_prompt, num_examples=num_init_neg_examples_adjusted)
                    
                    qualified_flag = False
                    current_num_attempts = 0
                    least_rel_examples = []
                    least_response_dump = {}
                    while qualified_flag is not True and current_num_attempts <= max_num_attempts:
                        try:
                            print(f"Attempt {current_num_attempts} for a gen template for one negative relation of relation {rel}")
                            current_num_attempts += 1
                            response = LLM_model(input={'task_id': task_id, 'messages': [{'role': 'user', 'content': deepcopy(prompt)}]})
                            
                            temp_rel_examples = parse_chat_response_to_examples(response=response, relation=f"!{rel}", num_examples=num_init_neg_examples_adjusted)
                            
                            response_dump = response.model_dump()
                            response_dump.update({'prompt_input': {'task_id': task_id, 'messages': [{'role': 'user', 'content': prompt}]}, 'relation': f"!{rel}", 'relative_id': f"{neg_rel_def_prompt_id}_{gen_template_id}"})
                            
                            
                            if len(temp_rel_examples) == num_init_neg_examples_adjusted:    
                                append_to_jsonl(
                                    data=response_dump,
                                    filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                                )
                                rel_example_list.extend(temp_rel_examples)
                                qualified_flag = True
                            else:
                                if abs(len(temp_rel_examples) - num_init_neg_examples_adjusted) <= 2 and len(temp_rel_examples) > len(least_rel_examples):
                                    least_rel_examples = temp_rel_examples
                                    least_response_dump = response_dump
                              
                        except openai.InternalServerError as e:
                            print("An Internal Server Error occurred when calling openai api:", e)
                    
                    
                    if not qualified_flag:
                        rel_example_list.extend(least_rel_examples)
                        append_to_jsonl(
                            data=least_response_dump,
                            filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                        )
                        print(f"Eventually LLM generation not qualified: Relation={rel} with #generated results={len(least_rel_examples)}. Use the sub-optimal generated results instead.")

                    task_id += 1
                
            rel_examples.append(rel_example_list)
            print(f'Parsed negative examples for {rel}')
            self.dataloader.print_div_raw_data_w_indices(example_list=rel_examples[-1], indent_char='|\t')
        
        torch.save(neg_def_prompts, os.path.join(llm_ckpt_folder, 'neg_def_prompts.pt'))
        return rel_examples



    def LLM_init_neg_generation_v1(self, rel_set, rel_def_prompt_list, save_suffix='', num_neg_rels_to_generate=3, num_init_neg_examples=20, llm_ckpt_folder=None):
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model.update_call_attributes(max_tokens=2048, seed=self.args.seed)
        

        max_num_attempts = 3
        
        neg_def_gen_template = Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\". Generate ${num_neg_rels} near-miss negative binary relation definitions in the same format (numbered from 1 to ${num_neg_rels}). First give your brief analysis (if any) and then list your numbered generated near-miss negative relation definitions (do not mix analysis and definitions).")
        
        gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\". One of its negative relations is defined by \"${neg_rel_def_prompt}\". Generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same negative relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Try not explicitly mention the positive relation. Do not overfit the pattern of the negative relation definition. Try as many different relation patterns or relation expressions as possible."),
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\". One of its negative relations is defined by \"${neg_rel_def_prompt}\". Generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same negative relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Try not explicitly mention the positive relation. Other content requirements: 1. Do not overfit the pattern of the negative relation definition. Try as many different relation patterns or relation expressions as possible. 2. Generate rich and informative related contexts before and after each entity."),
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\". One of its negative relations is defined by \"${neg_rel_def_prompt}\". Generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same negative relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Try not explicitly mention the positive relation. Other content requirements: 1. Do not overfit the pattern of the negative relation definition. Try as many different relation patterns or relation expressions as possible. 2. Generate rich and informative related contexts before and after each entity. 3. The relation patterns or relation expressions should be implicit or complicated")
        ]
        
        num_init_neg_examples_adjusted = math.ceil(num_init_neg_examples / len(gen_template_list) / num_neg_rels_to_generate)
        
        rel_examples = [] # list of list of examples with each sub list corresponds to each rel in rel_set
        
        task_id = 0
        for rel, rel_def_prompt in zip(rel_set, rel_def_prompt_list):
            rel_example_list = []
            
            # first generate neg relation definitions
            neg_rel_def_prompts = []
            neg_def_gen_prompt = neg_def_gen_template.substitute(pos_rel_def_prompt=rel_def_prompt, num_neg_rels=num_neg_rels_to_generate)
            
            qualified_flag = False
            current_num_attempts = 0
            least_neg_rel_def_prompts = []
            least_response_dump = {}
            while qualified_flag is not True and current_num_attempts <= max_num_attempts:
                try:
                    print(f"Attempt {current_num_attempts} for generating negative relation definitions for relation {rel}")
                    current_num_attempts += 1
                    
                    response = LLM_model(input={'task_id': task_id, 'messages': [{'role': 'user', 'content': deepcopy(neg_def_gen_prompt)}]})
                    
                    temp_neg_rel_def_prompts = parse_chat_response_to_def_prompts(response=response, num_def_prompts=num_neg_rels_to_generate)
                    
                    response_dump = response.model_dump()
                    response_dump.update({'prompt_input': {'task_id': task_id, 'messages': [{'role': 'user', 'content': neg_def_gen_prompt}]}, 'relation': f"{rel}->!{rel}",})
                    
                    if len(temp_neg_rel_def_prompts) == num_neg_rels_to_generate:
                        append_to_jsonl(
                            data=response_dump,
                            filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                        )
                        neg_rel_def_prompts = temp_neg_rel_def_prompts
                        qualified_flag = True
                    else:
                        if abs(len(temp_neg_rel_def_prompts) - num_neg_rels_to_generate) <= 2 and len(temp_neg_rel_def_prompts) > len(least_neg_rel_def_prompts):
                            least_neg_rel_def_prompts = temp_neg_rel_def_prompts
                            least_response_dump = response_dump
                except openai.InternalServerError as e:
                    print("An Internal Server Error occurred when calling openai api:", e)
            
            if not qualified_flag:
                neg_rel_def_prompts = least_neg_rel_def_prompts
                append_to_jsonl(
                    data=least_response_dump,
                    filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                )
                print(f"Eventually LLM generation not qualified: Relation={rel} with #generated results={len(least_neg_rel_def_prompts)}. Use the sub-optimal generated results instead.")
            
            print(f'Parsed negative relation definitions for {rel}')
            for neg_rel_def_prompt_to_print in neg_rel_def_prompts: print(f"\t-\t{neg_rel_def_prompt_to_print}")
            
            task_id += 1
            
            for neg_rel_def_prompt in neg_rel_def_prompts:
                for gen_template in gen_template_list:
                    prompt = gen_template.substitute(pos_rel_def_prompt=rel_def_prompt, neg_rel_def_prompt=neg_rel_def_prompt, num_examples=num_init_neg_examples_adjusted)
                    
                    qualified_flag = False
                    current_num_attempts = 0
                    least_rel_examples = []
                    least_response_dump = {}
                    while qualified_flag is not True and current_num_attempts <= max_num_attempts:
                        try:
                            print(f"Attempt {current_num_attempts} for a gen template for one negative relation of relation {rel}")
                            current_num_attempts += 1
                            response = LLM_model(input={'task_id': task_id, 'messages': [{'role': 'user', 'content': deepcopy(prompt)}]})
                            
                            temp_rel_examples = parse_chat_response_to_examples(response=response, relation=f"!{rel}", num_examples=num_init_neg_examples_adjusted)
                            
                            response_dump = response.model_dump()
                            response_dump.update({'prompt_input': {'task_id': task_id, 'messages': [{'role': 'user', 'content': prompt}]}, 'relation': f"!{rel}",})
                            
                            
                            if len(temp_rel_examples) == num_init_neg_examples_adjusted:    
                                append_to_jsonl(
                                    data=response_dump,
                                    filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                                )
                                rel_example_list.extend(temp_rel_examples)
                                qualified_flag = True
                            else:
                                if abs(len(temp_rel_examples) - num_init_neg_examples_adjusted) <= 2 and len(temp_rel_examples) > len(least_rel_examples):
                                    least_rel_examples = temp_rel_examples
                                    least_response_dump = response_dump
                                    
                        except openai.InternalServerError as e:
                            print("An Internal Server Error occurred when calling openai api:", e)
                    
                    
                    if not qualified_flag:
                        rel_example_list.extend(least_rel_examples)
                        append_to_jsonl(
                            data=least_response_dump,
                            filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                        )
                        print(f"Eventually LLM generation not qualified: Relation={rel} with #generated results={len(least_rel_examples)}. Use the sub-optimal generated results instead.")

                    task_id += 1
                
            rel_examples.append(rel_example_list)
            print(f'Parsed negative examples for {rel}')
            self.dataloader.print_div_raw_data_w_indices(example_list=rel_examples[-1], indent_char='|\t')
            
        return rel_examples


    def run_snowball(self, K=5, backup=50, test_div='test', unlabeled_corpus_div='distant', save_suffix='', deterministic=True):
        """

        Note: important assumption when using existing ckpts (the rel def prompt and the paraphrased prompts should be the same since we initialize the model and conduct unlabeled inference using the new prompts while the resumed ckpts have their old prompts)
        """
        
        # some hyperparams
        max_paraphrased_prompts = 0 # use at most this number of paraphrased rel def prompts
        num_init_pos_examples = self.args.num_init_pos_examples
        num_init_neg_examples = self.args.num_init_neg_examples
        buffer_factor = 10

        # define cache folders to ckpt and llm generation
        if save_suffix is None: 
            if self.args.run_neg_init_gen: extra_save_suffix = '_iniNegGen'
            else: extra_save_suffix = ''
            save_suffix = f'_{num_init_pos_examples}p{num_init_neg_examples}n{extra_save_suffix}'
            
        ckpt_sub_folder = f'snowball_ckpt{save_suffix}_seed{self.args.seed}/'
        llm_ckpt_folder = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, f'llm_{num_init_pos_examples}p_seed{self.args.seed}/')
        llm_neg_ckpt_folder = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, f'llm_{self.args.num_init_neg_examples_to_generate}n_iniNegGen_seed{self.args.seed}/')
        
        
        # sample relations and related instance ids for evaluation and few-shot sampling for each iteration
        rel_set, rel2Kshot_locIds, eval_locIds, rel2backup_locIds = self.dataloader.sample_and_gather_indices(div=test_div, K=K, backup=backup, deterministic=deterministic)
        

        # initialize LLM cache folder&files
        if not os.path.exists(llm_ckpt_folder): os.makedirs(llm_ckpt_folder)
        rels_pos_examples_ckpt = os.path.join(llm_ckpt_folder, f'rel_init_pos.pt')
        if os.path.exists(rels_pos_examples_ckpt):
            rels_pos_examples = torch.load(rels_pos_examples_ckpt)
        else:
            rels_pos_examples = self.LLM_init_pos_generation_parallel(rel_set=rel_set, rel_def_prompt_list=[self.dataloader.rel_info[r]["typed_desc_prompt"] for r in rel_set], save_suffix=f'_{num_init_pos_examples}p/', num_init_pos_examples=num_init_pos_examples, llm_ckpt_folder=llm_ckpt_folder)
            torch.save(rels_pos_examples, rels_pos_examples_ckpt)
        
        # check of the positive seeds' relations match with the relations in order
        for i, rel in enumerate(rel_set): assert rel == rels_pos_examples[i][0]['relation'], "The rel_set does not match with init rels_pos_examples! Need re-init pos examples and set up the seeds!"



        rels_dev_pos_examples_ckpt = os.path.join(llm_ckpt_folder, f'rel_dev_pos_0.pt')
        if os.path.exists(rels_dev_pos_examples_ckpt):
            rels_dev_pos_examples = torch.load(rels_dev_pos_examples_ckpt)
        else:
            llm_ckpt_dev_folder = os.path.join(llm_ckpt_folder, 'dev')
            os.makedirs(llm_ckpt_dev_folder, exist_ok=True)
            rels_dev_pos_examples = self.LLM_init_pos_generation_parallel(rel_set=rel_set, rel_def_prompt_list=[self.dataloader.rel_info[r]["typed_desc_prompt"] for r in rel_set], save_suffix=f'_{num_init_pos_examples}p/', num_init_pos_examples=num_init_pos_examples, llm_ckpt_folder=llm_ckpt_dev_folder)
            torch.save(rels_dev_pos_examples, rels_dev_pos_examples_ckpt)
        
        for i, rel in enumerate(rel_set): assert rel == rels_dev_pos_examples[i][0]['relation'], "The rel_set does not match with init rels_dev_pos_examples! Need re-init pos dev examples and set up the seeds!"
            
            

        if (not os.path.exists(llm_neg_ckpt_folder)) and self.args.run_neg_init_gen: os.makedirs(llm_neg_ckpt_folder)
        rels_neg_examples = None
        if self.args.run_neg_init_gen:
            rels_neg_examples_ckpt = os.path.join(llm_neg_ckpt_folder, f'rel_init_neg.pt')
            if os.path.exists(rels_neg_examples_ckpt):
                rels_neg_examples = torch.load(rels_neg_examples_ckpt)
            else:
                rels_neg_examples = self.LLM_init_neg_generation(rel_set=rel_set, rel_def_prompt_list=[self.dataloader.rel_info[r]["typed_desc_prompt"] for r in rel_set], save_suffix=f'_{self.args.num_init_neg_examples_to_generate}n/', num_neg_rels_to_generate=self.args.num_init_neg_rels_to_generate, num_init_neg_examples=self.args.num_init_neg_examples_to_generate, llm_ckpt_folder=llm_neg_ckpt_folder)
                torch.save(rels_neg_examples, rels_neg_examples_ckpt)

            assert rels_neg_examples is not None and len(rels_neg_examples) >= 0
        
        # create the ckpt dir and tensorboad writer
        os.makedirs(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, 'tensorboard/'))


        hist_evaluation_results = {
            'test_div': test_div,
            'unlabeled_corpus_div': unlabeled_corpus_div,
            'args': self.args,
            'rel_set': rel_set,
            'rel2Kshot_locIds': rel2Kshot_locIds,
            'eval_locIds': eval_locIds,
            'rel2backup_locIds': rel2backup_locIds,
            'rel': [],
            'rel_def_prompt': [],
            'rel_def_paraphrased_prompts': [],
            'pos_examples': [],
            'neg_examples': [],
            'rel_hist_evaluation_results': [],
            'rel_unlabeled_inference_results': [],
            'train_inference_ckpt_exists': [],
        }

        for rel_idx, rel in enumerate(rel_set):
            
            print(f"\n\n\n===Conducting Snowball on relation: {rel} ({self.dataloader.rel_info[rel]['relation_name']})===")

            # given relation definition of this relation: in the form like "<ENT1> is the primary topic of <ENT0> (a work)" where <ENT0> corresponds to head entity and <ENT1> corresponds to tail entity
            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            # list of strings with each string as the prompt paraphrased from the relation definition prompt (in the same format)
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
            # visualize
            print(f"Relation definition prompt for {rel}: {rel_def_prompt}")
            for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                print(f"|\tParaphrased prompt {i} for {rel}:\t{rel_def_paraphrased_prompt}")


            rel_pos_examples, rel_neg_examples = [], []
            # obtain generated pos samples using LLM
            rel_pos_examples = rels_pos_examples[rel_idx] # list of examples
            print("LLM generated initial positive examples: ")
            self.dataloader.print_div_raw_data_w_indices(div_list=None, local_indices=None, indent_char='|\t', example_list=rel_pos_examples)

            rel_dev_pos_examples = rels_dev_pos_examples[rel_idx]
            
            if self.args.run_neg_init_gen:
                rel_neg_examples = deepcopy(rels_neg_examples[rel_idx])
                if num_init_neg_examples > self.args.num_init_neg_examples_to_generate:
                    num_init_neg_examples_to_sample = num_init_neg_examples - self.args.num_init_neg_examples_to_generate
                    if deterministic: random.seed(min(self.args.seed, 1) * (rel_idx + 1))
                    rel_init_neg_indices = random.sample(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])), num_init_neg_examples_to_sample * buffer_factor)
                    rel_neg_examples_buffered = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_init_neg_indices)
                    rel_neg_examples.extend(select_valid_distinct_examples(ref_examples=rel_neg_examples, buffered_examples=rel_neg_examples_buffered, target_num_samples=num_init_neg_examples_to_sample))
                    print(f"LLM generated initial negative examples (first {len(rel_neg_examples) - num_init_neg_examples_to_sample}) and randomly sampled initial negative examples (last {num_init_neg_examples_to_sample}): ")
                else:
                    print(f"LLM generated initial negative examples ({len(rel_neg_examples)}) and no random sampling for initial negative examples. Since  num_init_neg_examples ({num_init_neg_examples}) <= self.args.num_init_neg_examples_to_generate ({self.args.num_init_neg_examples_to_generate})")
            else:
                # randomly sample negative samples from unlabeled corpus
                if deterministic: random.seed(min(self.args.seed, 1) * (rel_idx + 1))
                rel_init_neg_indices = random.sample(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])), num_init_neg_examples * buffer_factor)
                rel_neg_examples_buffered = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_init_neg_indices)
                rel_neg_examples = select_valid_distinct_examples(ref_examples=[], buffered_examples=rel_neg_examples_buffered, target_num_samples=num_init_neg_examples)
                print("Randomly sampled initial negative examples from unlabeled corpus: ")
                
            self.dataloader.print_div_raw_data_w_indices(div_list=None, local_indices=None, indent_char='|\t', example_list=rel_neg_examples)
            
            rel_init_dev_neg_indices = random.sample(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])), num_init_neg_examples * buffer_factor)
            rel_dev_neg_examples_buffered = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_init_dev_neg_indices)
            rel_dev_neg_examples = select_valid_distinct_examples(ref_examples=rel_neg_examples, buffered_examples=rel_dev_neg_examples_buffered, target_num_samples=num_init_neg_examples)
            
            rel_dev_pos_neg_examples = rel_dev_pos_examples + rel_dev_neg_examples
            rel_dev_pos_neg_assigned_labels = [torch.tensor([1], dtype=torch.long)] * len(rel_dev_pos_examples) + [torch.tensor([0], dtype=torch.long)] * len(rel_dev_neg_examples)
            

            rel_pos_neg_examples = rel_pos_examples + rel_neg_examples
            print(f"Training RE model with {len(rel_pos_examples)} positive examples and {len(rel_neg_examples)} negative examples in total.")
            rel_pos_neg_assigned_labels = [torch.tensor([1], dtype=torch.long)] * len(rel_pos_examples) + [torch.tensor([0], dtype=torch.long)] * len(rel_neg_examples)
            rel_NLI_model = NLIBasedSimilarityModel(args=self.args, tokenizer=self.dataloader.tokenizer, target_relation=rel, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts)
            test_div2local_indices = {
                test_div: eval_locIds,
            }
            rel_hist_evaluation_results, ckpt_exists, rel_NLI_model = self.NLIBased_Optim_w_PosNeg_rel(rel_NLI_model=rel_NLI_model, target_rel=rel, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, div2local_indices=None, div2assigned_labels=None, test_div2local_indices=test_div2local_indices, ckpt_sub_folder=ckpt_sub_folder, train_examples=rel_pos_neg_examples, train_assigned_labels=rel_pos_neg_assigned_labels, SummaryWriter=writer, extra_info_to_save={
                'pos_examples_0': rel_pos_examples, 'neg_examples_0': rel_neg_examples,
                'pos_dev_examples_0': rel_dev_pos_examples, 'neg_dev_examples_0': rel_dev_neg_examples ### FOR DEV SET ###
            }, dev_examples=rel_dev_pos_neg_examples, dev_assigned_labels=rel_dev_pos_neg_assigned_labels, )  ### FOR DEV SET ###


            if ckpt_exists:
                print(f"===Ckpt found for relation: {rel}. Skipped its optimization round. Use the ckpt as an approximation instead.===")
                rel_pos_examples = rel_hist_evaluation_results['pos_examples_0']
                rel_neg_examples = rel_hist_evaluation_results['neg_examples_0']

            hist_evaluation_results['rel'].append(rel)
            hist_evaluation_results['rel_def_prompt'].append(rel_def_prompt)
            hist_evaluation_results['rel_def_paraphrased_prompts'].append(rel_def_paraphrased_prompts)
            hist_evaluation_results['pos_examples'].append(rel_pos_examples)
            hist_evaluation_results['neg_examples'].append(rel_neg_examples)
            hist_evaluation_results['rel_hist_evaluation_results'].append(rel_hist_evaluation_results)
            hist_evaluation_results['train_inference_ckpt_exists'].append(ckpt_exists)
            

        for epoch_idx in self.args.save_epochs:
            print(f"Results (at epoch={epoch_idx}) averaged over all relations: P: {np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")

        print(f"Chosen results averaged over all relations: P: {np.mean([i['chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")


        for epoch_idx in range(self.args.num_train_epochs):
            avg_precision = np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged f1", avg_f1, epoch_idx)


            avg_precision = np.mean([i['dev_precision'][i['dev_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['dev_recall'][i['dev_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['dev_f1'][i['dev_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged dev precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev f1", avg_f1, epoch_idx)


        dev_local_chosen_precision = np.mean([i['dev_chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_recall = np.mean([i['dev_chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_f1 = np.mean([i['dev_chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        writer.add_scalar(f"All Relations/dev local chosen precision", dev_local_chosen_precision, 0)
        writer.add_scalar(f"All Relations/dev local chosen recall", dev_local_chosen_recall, 0)
        writer.add_scalar(f"All Relations/dev local chosen f1", dev_local_chosen_f1, 0)


        gc.collect()
        torch.cuda.empty_cache() 

        for rel_idx, rel in enumerate(rel_set):
            rel_ckpt_exists = hist_evaluation_results['train_inference_ckpt_exists'][rel_idx]
            rel_hist_evaluation_results = hist_evaluation_results['rel_hist_evaluation_results'][rel_idx]
            
            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
            
            
            # if do not run further snowball, return
            if self.args.run_snowball == False: 
                if os.path.exists(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')):
                    print(f"===run_snowball is False, but found cached unlabeled corpus inference results. Load cached unlabeled corpus inference results to hist_evaluation_results.===")
                    hist_evaluation_results['rel_unlabeled_inference_results'].append(torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')))
                else:
                    print(f"===run_snowball is False, didn't found cached unlabeled corpus inference results. Appending an empty python dictionary to hist_evaluation_results.===")
                    hist_evaluation_results['rel_unlabeled_inference_results'].append({})

                continue

            if ckpt_exists and os.path.exists(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')):
                print(f"===Ckpt found for relation: {rel}. Skipped its optimization round and its unlabeled corpus inference round. Use cached unlabeled corpus inference instead.===")
                hist_evaluation_results['rel_unlabeled_inference_results'].append(torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')))
                continue


            gc.collect()
            torch.cuda.empty_cache()

            rel_NLI_model = NLIBasedSimilarityModel(args=self.args, tokenizer=self.dataloader.tokenizer, target_relation=rel, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts)
            print('Before inference on unlabeled corpus, model is on device=', next(rel_NLI_model.parameters()).device)


            # update rel_NLI_model's parameters with the chosen trained ckpt
            chosen_model_state_dict_path = rel_hist_evaluation_results['dev_chosen_rel_NLI_model_state_dict_path']


            # get all the unlabeled corpus examples and conduct inference over them
            rel_unlabeled_indices = list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])))
            rel_unlabeled_examples = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_unlabeled_indices)

            free_port = find_free_port(start_port=self.args.dist_port, end_port=self.args.dist_port+200)
            print(f"Found free port at {free_port}, switching from {self.args.dist_port}")
            self.args.dist_port = free_port

            manager = mp.Manager()
            shared_results = manager.list()
            mp.spawn(NLIBased_Inference, args=(self.args.num_gpus, self.args, chosen_model_state_dict_path, self.dataloader, rel_unlabeled_examples, rel, rel_def_prompt, rel_def_paraphrased_prompts, shared_results, ckpt_sub_folder, 0.5), nprocs=self.args.num_gpus)
            rel_unlabeled_inference_results = list(shared_results)[0]

            hist_evaluation_results['rel_unlabeled_inference_results'].append(rel_unlabeled_inference_results)
            
        
        self.save_model(
            os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_all.pt'),
            hist_evaluation_results,
        )


        return hist_evaluation_results

            
    def LLM_followup_pos_generation_parallel(self, rel_set, representative_relation_patterns_examples, init_llm_prompts, num_followup_pos_examples=10, llm_ckpt_folder=None, sliding_window_size=10, enable_sliding_window=True, feedback_w_scores=False, representative_relation_patterns_examples_scores=None):
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat_parallel_pos.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model.update_call_attributes(max_tokens=4096, seed=self.args.seed, temperature=self.args.temperature)
    
        max_num_attempts = 3
        

        gen_template = Template("Sampled examples which are predicted as positive by my relation extraction model are:\n\n${feedback_examples}\n\nBased on these predicted examples and your previously generated examples, generate ${num_examples} additional examples (numbered from 1 to ${num_examples}) expressing the same pre-defined relation: \"${rel_def_prompt}\". Other requirements are:\n\n1. Identify what relation patterns have been learnt by my model and what relation patterns have been covered by your previously generated examples. Your newly generated examples should have different and diverse relation patterns.\n2. Identify model's bias from the sampled predicted examples which do not express the correct relation definition and your newly generated examples should try to mitigate the bias.\n3. If the sampled predicted examples are uninformative, focus on the dialogue history, especially examples that were previously generated, to generate new examples with different and more diverse patterns.")  
        
        
        if self.args.run_LLM_json_parser_ex: 
            chat_response_to_examples_parser = lambda response, relation, num_examples: parse_chat_response_to_examples_LLM_json_parser(response=response, relation=relation, num_examples=num_examples, llm_model_ckpt=self.args.llm_model_ckpt_parser)
        else:
            chat_response_to_examples_parser = parse_chat_response_to_examples
        
        task_id = None
        input_list = []
        input_list_x_elt = []
        for rel_id, rel in enumerate(rel_set):
            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            
            rel_init_llm_prompts = []
            for init_llm_prompt in init_llm_prompts:
                if init_llm_prompt['relation'] == rel: rel_init_llm_prompts.append(init_llm_prompt)
            
            rel_representative_relation_patterns_examples = representative_relation_patterns_examples[rel_id]
            if feedback_w_scores: rel_representative_relation_patterns_examples_scores = representative_relation_patterns_examples_scores[rel_id]
            if not enable_sliding_window:
                feedback_examples_to_print = []
                for ex_id, ex in enumerate(rel_representative_relation_patterns_examples):
                    if feedback_w_scores:
                        feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True) + f"(Predicted probability={rel_representative_relation_patterns_examples_scores[ex_id]:.2f})")
                    else:
                        feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True))
                feedback_examples_to_print_string = "\n\n".join(feedback_examples_to_print)
                    
            num_examples_per_prompt = math.ceil(num_followup_pos_examples / len(rel_init_llm_prompts))
            
            for rel_init_llm_prompt_id, rel_init_llm_prompt in enumerate(rel_init_llm_prompts):
                if enable_sliding_window:
                    feedback_examples_to_print = []
                    for ex_id, ex in enumerate(rel_representative_relation_patterns_examples[rel_init_llm_prompt_id * sliding_window_size: rel_init_llm_prompt_id * sliding_window_size + sliding_window_size]):
                        if feedback_w_scores:
                            feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True) + f"(Predicted probability={rel_representative_relation_patterns_examples_scores[rel_init_llm_prompt_id * sliding_window_size: rel_init_llm_prompt_id * sliding_window_size + sliding_window_size][ex_id]:.2f})")
                        else:
                            feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True))
                    feedback_examples_to_print_string = "\n\n".join(feedback_examples_to_print)
                
                
                if '${rel_def_prompt}' in gen_template.template: new_turn_content = gen_template.substitute(feedback_examples=feedback_examples_to_print_string, num_examples=num_examples_per_prompt, rel_def_prompt=rel_def_prompt)
                else: new_turn_content = gen_template.substitute(feedback_examples=feedback_examples_to_print_string, num_examples=num_examples_per_prompt)
                
                task_id = deepcopy(rel_init_llm_prompt['prompt_input']['task_id'])
                
                conversations = deepcopy(rel_init_llm_prompt['prompt_input']['messages'])
                conversations.append({
                    'role': deepcopy(rel_init_llm_prompt['choices'][0]['message']['role']),
                    'content': deepcopy(rel_init_llm_prompt['choices'][0]['message']['content']),
                })
                conversations.append({
                    'role': 'user',
                    'content': deepcopy(new_turn_content),
                })

                
                input_list.append({
                    'task_id': task_id,
                    'messages': conversations,
                })
                input_list_x_elt.append(f"{rel_id}-{rel_init_llm_prompt_id}_{rel_init_llm_prompt['relative_id']}")
        
        
        output_list = LLM_model.async_calls(input_list=input_list)
        current_best_output_list = deepcopy(output_list)
        rels_pos_examples_followup = []
        
        input_list_to_regen = []
        input_list_to_regen_x_elt = []
        
        current_num_attempts = 0
        for input_idx, (input_x_elt, output_response) in enumerate(zip(input_list_x_elt, output_list)):
            rel_id = int(input_x_elt.split('_')[0].split('-')[0])
            rel_init_llm_prompt_id = int(input_x_elt.split('_')[0].split('-')[1])
            rel = rel_set[rel_id]
            
            
            temp_rel_examples = chat_response_to_examples_parser(response=deepcopy(output_response), relation=rel, num_examples=deepcopy(num_examples_per_prompt))
            if len(temp_rel_examples) == num_examples_per_prompt:
                rels_pos_examples_followup.append(deepcopy(temp_rel_examples))
            else:
                input_list_to_regen.append(deepcopy(input_list[input_idx]))
                input_list_to_regen_x_elt.append(deepcopy(input_x_elt))
                rels_pos_examples_followup.append(None)
                
                
        while (None in rels_pos_examples_followup) and current_num_attempts <= max_num_attempts:
            print(f"Current attempt: {current_num_attempts} with #invalid inputs to regenerate={rels_pos_examples_followup.count(None)}")
            assert rels_pos_examples_followup.count(None) == len(input_list_to_regen)
            
            output_list_to_regen = LLM_model.async_calls(input_list=input_list_to_regen)
            
            input_list_to_regen_temp = []
            input_list_to_regen_x_elt_temp = []
            
            for input_to_regen_idx, (input_x_elt_to_regen, output_response_to_regen) in enumerate(zip(input_list_to_regen_x_elt, output_list_to_regen)):  
                rel_id = int(input_x_elt_to_regen.split('_')[0].split('-')[0])
                rel_init_llm_prompt_id = int(input_x_elt_to_regen.split('_')[0].split('-')[1])
                rel = rel_set[rel_id]

                absolute_idx = input_list_x_elt.index(input_x_elt_to_regen)
                
                
                temp_rel_examples = chat_response_to_examples_parser(response=deepcopy(output_response_to_regen), relation=rel, num_examples=deepcopy(num_examples_per_prompt))
                
                if len(temp_rel_examples) == num_examples_per_prompt:
                    assert rels_pos_examples_followup[absolute_idx] is None
                    rels_pos_examples_followup[absolute_idx] = deepcopy(temp_rel_examples)
                    current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
                else:
                    input_list_to_regen_temp.append(deepcopy(input_list_to_regen[input_to_regen_idx]))
                    input_list_to_regen_x_elt_temp.append(deepcopy(input_x_elt_to_regen))
                    
                    if abs(len(temp_rel_examples) - num_examples_per_prompt) <= 2 and len(temp_rel_examples) > len(chat_response_to_examples_parser(response=deepcopy(current_best_output_list[absolute_idx]), relation=rel, num_examples=deepcopy(num_examples_per_prompt))):
                        current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)

            input_list_to_regen = deepcopy(input_list_to_regen_temp)
            input_list_to_regen_x_elt = deepcopy(input_list_to_regen_x_elt_temp)
            
            current_num_attempts += 1
        
        rels_pos_examples_followup_output = []
        for rels_pos_examples_followup_idx, (input_x_elt, rels_pos_examples_followup_list) in enumerate(zip(input_list_x_elt, rels_pos_examples_followup)):
            rel_id = int(input_x_elt.split('_')[0].split('-')[0])
            rel_init_llm_prompt_id = int(input_x_elt.split('_')[0].split('-')[1])
            rel = rel_set[rel_id]
            
            if rel_init_llm_prompt_id == 0: rels_pos_examples_followup_output.append([])
            
            if rels_pos_examples_followup_list is not None:
                rels_pos_examples_followup_output[rel_id].extend(rels_pos_examples_followup_list)
            else:
                rels_pos_examples_followup_output[rel_id].extend(chat_response_to_examples_parser(response=deepcopy(current_best_output_list[rels_pos_examples_followup_idx]), relation=rel, num_examples=deepcopy(num_examples_per_prompt)))
                    
            
            response = current_best_output_list[rels_pos_examples_followup_idx]
            response_dump = response.model_dump()
            response_dump.update({'prompt_input': input_list[rels_pos_examples_followup_idx], 'relation': rel, 'relative_id': input_x_elt})
            append_to_jsonl(
                data=response_dump,
                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat_pos.jsonl'),
            )
            
        for rel_id, rel in enumerate(rel_set):
            print(f'Parsed examples for {rel}')
            self.dataloader.print_div_raw_data_w_indices(example_list=rels_pos_examples_followup_output[rel_id], indent_char='|\t')
        
        return rels_pos_examples_followup_output

  
    def LLM_followup_pos_generation(self, rel_set, representative_relation_patterns_examples, init_llm_prompts, num_followup_pos_examples=10, llm_ckpt_folder=None, sliding_window_size=10, enable_sliding_window=True):
        """
        Generate positive examples for relations given the feedback examples from unlabeled corpus. 

        This function's generated files can be reused by next time calling the same function to continue the conversation.
        """
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat_pos.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model.update_call_attributes(max_tokens=4096, seed=self.args.seed)
        
        max_num_attempts = 3
        
        gen_template = Template("Typical examples predicted by my relation extraction model are:\n\n${feedback_examples}\n\nBased on these predicted examples and your previously generated examples, generate ${num_examples} additional examples (numbered from 1 to ${num_examples}) expressing the pre-defined relation. Other requirements are: 1. Identify what relation patterns have been learnt by my model or covered by your previously generated examples and your newly generated examples should have different and diverse relation patterns. 2. Identify model's bias from the predicted examples which do not express the correct relation definition and your newly generated examples should try to mitigate the bias.")

        rels_pos_examples_followup = []

        task_id = None
        for rel_id, rel in enumerate(rel_set):
            rel_example_list = []

            rel_init_llm_prompts = []
            for init_llm_prompt in init_llm_prompts:
                if init_llm_prompt['relation'] == rel: rel_init_llm_prompts.append(init_llm_prompt)
                
                
            rel_representative_relation_patterns_examples = representative_relation_patterns_examples[rel_id]
            if not enable_sliding_window:
                feedback_examples_to_print = []
                for ex_id, ex in enumerate(rel_representative_relation_patterns_examples):
                    feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True))
                feedback_examples_to_print_string = "\n\n".join(feedback_examples_to_print)
            

            num_examples_per_prompt = math.ceil(num_followup_pos_examples / len(rel_init_llm_prompts))
            
            for rel_init_llm_prompt_id, rel_init_llm_prompt in enumerate(rel_init_llm_prompts):
                if enable_sliding_window:
                    feedback_examples_to_print = []
                    for ex_id, ex in enumerate(rel_representative_relation_patterns_examples[rel_init_llm_prompt_id * sliding_window_size:rel_init_llm_prompt_id * sliding_window_size + sliding_window_size]):
                        feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True))
                    feedback_examples_to_print_string = "\n\n".join(feedback_examples_to_print)
            
                
                new_turn_content = gen_template.substitute(feedback_examples=feedback_examples_to_print_string, num_examples=num_examples_per_prompt)
                
                
                qualified_flag = False
                current_num_attempts = 0
                least_rel_examples = []
                least_response_dump = {}
                
                while qualified_flag is not True and current_num_attempts <= max_num_attempts:
                    try:
                        print(f"Attempt {current_num_attempts} for a single init llm prompt of relation {rel}")
                        current_num_attempts += 1
                        
                        task_id = deepcopy(rel_init_llm_prompt['prompt_input']['task_id'])
                        # most recent previous query to llm
                        conversations = deepcopy(rel_init_llm_prompt['prompt_input']['messages'])
                        # append the most recent previous answer from llm
                        conversations.append({
                            'role': deepcopy(rel_init_llm_prompt['choices'][0]['message']['role']),
                            'content': deepcopy(rel_init_llm_prompt['choices'][0]['message']['content']),
                        })
                        # append the current query to llm
                        conversations.append({
                            'role': 'user',
                            'content': deepcopy(new_turn_content),
                        })

                        response = LLM_model(input={'task_id': task_id, 'messages': conversations})   
                        temp_rel_examples = parse_chat_response_to_examples(response=response, relation=rel, num_examples=num_examples_per_prompt)
                        response_dump = response.model_dump()
                        response_dump.update({'prompt_input': {'task_id': task_id, 'messages': conversations}, 'relation': rel,})
                        
                        if len(temp_rel_examples) == num_examples_per_prompt:
                            append_to_jsonl(
                                data=response_dump,
                                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat_pos.jsonl'),
                            )
                            rel_example_list.extend(temp_rel_examples)
                            
                            qualified_flag = True
                        else:
                            if abs(len(temp_rel_examples) - num_examples_per_prompt) <= 2 and len(temp_rel_examples) > len(least_rel_examples):
                                least_rel_examples = temp_rel_examples
                                least_response_dump = response_dump
                                
                    except openai.InternalServerError as e:
                        print("An Internal Server Error occurred when calling openai api:", e)
        
        
                if not qualified_flag:
                    rel_example_list.extend(least_rel_examples)
                    append_to_jsonl(
                        data=least_response_dump,
                        filename=os.path.join(llm_ckpt_folder, f'gpt4_chat_pos.jsonl'),
                    )
                    print(f"Eventually LLM generation not qualified: Relation={rel} with #generated results={len(least_rel_examples)}. Use the sub-optimal generated results instead.")

            rels_pos_examples_followup.append(rel_example_list)
            print(f'Parsed examples for {rel}')
            self.dataloader.print_div_raw_data_w_indices(example_list=rels_pos_examples_followup[-1], indent_char='|\t')
            

        return rels_pos_examples_followup
    


    def _get_rels_pos_examples_followup(self, rels_pos_examples_followup_ckpt, llm_ckpt_folder_step0, llm_ckpt_folder_step1, rel_set, max_paraphrased_prompts, hist_evaluation_results, unlabeled_inference_results, unlabeled_corpus_div, num_follow_pos_examples, new_threshold=0.85, top_N=50000):
        """
        if the rels_pos_examples_followup_ckpt exists (which contains the followup pos examples for all relations after LLM receiving the feedback examples), do nothing. Otherwise, first mine feedback examples from the inference results over the unlabeled corpus (representative_relation_patterns_examples refers to such feedback examples and this function will first check if cached feedback examples exist to reuse) and then use those feedback examples to further prompt LLM to generate followup examples which will be saved to rels_pos_examples_followup_ckpt.

        Args:
            rels_pos_examples_followup_ckpt: existing ckpt of the followup examples for all relations. 
            llm_ckpt_folder_step0: ckpt folder for the initially generated positive examples
            llm_ckpt_folder_step1: path for storing the feedback examples used to prompt LLM to generate followup examples (feedback examples are/should be stored in os.path.join(llm_ckpt_folder_step1, 'feedback_examples.pt'))

        Returns: 
            None. Need to load from the saved file instead of getting results from this function directly.
        """
        if not os.path.exists(rels_pos_examples_followup_ckpt):
            init_llm_prompts = load_jsonl(filename=os.path.join(llm_ckpt_folder_step0, 'gpt4_chat.jsonl'))

            # https://spacy.io/usage/processing-pipelines#_title for more info on spacy nlp.pipe
            spacy.prefer_gpu()
            # nlp = spacy.load("en_core_web_lg")
            nlp = spacy.load("en_core_web_trf")
            # number of available cores/processors
            n_cores = multiprocessing.cpu_count()


            representative_relation_patterns_examples = []
            representative_relation_patterns_examples_ckpt = os.path.join(llm_ckpt_folder_step1, 'feedback_examples.pt')
            if os.path.exists(representative_relation_patterns_examples_ckpt):
                representative_relation_patterns_examples = torch.load(representative_relation_patterns_examples_ckpt)

            feedback_intermediate_results = {
                'relation_patterns': [],
                'relation_patterns_examples': [],
                'relation_patterns_w_masks': [],
                'relation_patterns_w_masks_tagged': [],
                'cluster_assignment': [],
                'clustered_data_indices_sorted': [],
            }

            for rel_id, rel in enumerate(rel_set):
                if os.path.exists(representative_relation_patterns_examples_ckpt):
                    print(f"Cached feedback examples file exists. Skipping mining process for {rel} ({self.dataloader.rel_info[rel]['relation_name']})...")
                    print("Feedback examples: ")
                    self.dataloader.print_div_raw_data_w_indices(example_list=representative_relation_patterns_examples[rel_id], indent_char='|\t')
                    continue

                print(f"\n\n\n===Mining relation pattern feedbacks to LLM: {rel} ({self.dataloader.rel_info[rel]['relation_name']})===")

                # given relation definition of this relation: in the form like "<ENT1> is the primary topic of <ENT0> (a work)" where <ENT0> corresponds to head entity and <ENT1> corresponds to tail entity
                rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
                # list of strings with each string as the prompt paraphrased from the relation definition prompt (in the same format)
                rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
                # visualize
                print(f"    Relation definition prompt for {rel}: {rel_def_prompt}")
                for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                    print(f"        Paraphrased prompt {i} for {rel}:\t{rel_def_paraphrased_prompt}")


                rel_hist_evaluation_results = hist_evaluation_results['rel_hist_evaluation_results'][rel_id]
                rel_unlabeled_inference_results = unlabeled_inference_results[rel_id]
                assert is_ascending(array=rel_unlabeled_inference_results['inference_relative_ids'])
                assert rel == rel_hist_evaluation_results['rel']

                
                print(f"Setting the threshold={new_threshold} and setting top_N={top_N} for selecting positive instances from unlabled corpus")
                rel_unlabeled_pred_probs = rel_unlabeled_inference_results['pred_probs']
                rel_unlabeled_inference_relative_ids = rel_unlabeled_inference_results['inference_relative_ids']
                count_above_threshold = np.sum(rel_unlabeled_pred_probs[:, 1] >= new_threshold)
                if count_above_threshold < top_N: 
                    print("Number of pred probs >= new_threshold is smaller than top_N. Adjusting top_N to this number.")
                    top_N = count_above_threshold
                sorted_indices = np.argsort(rel_unlabeled_pred_probs[:, 1])[::-1] # numpy array of (|unlabeled corpus|, ), indicating the indices from largest pos pred prob to least
                rel_unlabeled_pred_probs_sorted = rel_unlabeled_pred_probs[sorted_indices, :] # numpy array of (|unlabeled corpus|, 2), with order from largest pos pred prob to least 
                rel_unlabeled_inference_relative_ids_sorted = rel_unlabeled_inference_relative_ids[sorted_indices] # numpy array of (|unlabeled corpus|, ), indicating the indices to the unlabeled corpus samples. 
                top_N_unlabeled_indices = rel_unlabeled_inference_relative_ids_sorted[:top_N]
                rel_pos_examples_unlabeled = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=top_N_unlabeled_indices)
                rel_pos_examples_snowball = remove_duplicated_invalid_examples(examples=rel_pos_examples_unlabeled)

                preprocessed_docs = list(nlp.pipe([" ".join(unlabeled_ex['tokens']) for unlabeled_ex in rel_pos_examples_snowball], batch_size=2048))

                h_t_unconnected_count = 0 # counting the number of examples where head ent and tail ent are not in the same dep parsing tree
                relation_patterns = []
                relation_patterns_examples = []
                relation_patterns_w_masks = []
                relation_patterns_w_masks_tagged = []
                rel_representative_relation_patterns_examples = []

                
                for i, unlabeled_ex in enumerate(tqdm(rel_pos_examples_snowball, desc='Examples')):
                    # print(i)
                    doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path = get_shortest_dep_path_v3(word_list=unlabeled_ex['tokens'], ht_word_spans=[[unlabeled_ex['h'][2][0][0], unlabeled_ex['h'][2][0][-1]], [unlabeled_ex['t'][2][0][0], unlabeled_ex['t'][2][0][-1]]], doc_given=preprocessed_docs[i])
                    
                    if h_doc_span_path is None: 
                        h_t_unconnected_count += 1
                        continue
                    

                    h_dep_t_string = " ".join(linearize_ent_dep_path(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, add_tags=False, add_h_t=True))
                    h_dep_t_string_w_mask = " ".join(linearize_ent_dep_path_w_mask(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, mask_h_t=False, mask_token='<mask>', tag_h_t=False))

                    h_dep_t_string_w_mask_tagged = " ".join(linearize_ent_dep_path_w_mask(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, mask_h_t=False, mask_token='<mask>', tag_h_t=True))

                    relation_patterns.append(h_dep_t_string)
                    relation_patterns_examples.append(unlabeled_ex)
                    relation_patterns_w_masks.append(h_dep_t_string_w_mask)
                    relation_patterns_w_masks_tagged.append(h_dep_t_string_w_mask_tagged)

                relation_patterns_embs = get_SBERT_embedding(sent_list=relation_patterns_w_masks, sbert_model='all-mpnet-base-v2', batch_size=768, normalize_embeddings=True)


                cluster_centroids, cluster_assignment, clustered_data_indices, clustered_dists, dist = get_KMeans_clusters(data_embeddings=relation_patterns_embs, seed=self.args.seed, n_clusters=10)
                clustered_data_indices_sorted = []
                for centroid_idx in range(len(clustered_data_indices)):
                    assert len(clustered_data_indices[centroid_idx]) == len(clustered_dists[centroid_idx])
                    clustered_data_indices_sorted.append([])
                    sorted_indices = np.argsort(clustered_dists[centroid_idx]) # should be reversed of using HDBSCAN
                    for j in sorted_indices: clustered_data_indices_sorted[-1].append(clustered_data_indices[centroid_idx][j])
                for centroid_idx in range(len(clustered_data_indices_sorted)):
                    rel_representative_relation_patterns_examples.append(relation_patterns_examples[clustered_data_indices_sorted[centroid_idx][0]])


                for centroid_idx, clustered_data_ids in enumerate(clustered_data_indices_sorted):
                    print(f'\nCluster {centroid_idx} and its top examples')
                    for j in clustered_data_ids[:20]:
                        print("     Pattern: " + relation_patterns_w_masks_tagged[j])
                        print("     Example:" + get_so_tagged_example(example=relation_patterns_examples[j], compact=True))
                        print('--' * 30)
                    print("\n")


                feedback_intermediate_results['relation_patterns'].append(relation_patterns)
                feedback_intermediate_results['relation_patterns_examples'].append(relation_patterns_examples)
                feedback_intermediate_results['relation_patterns_w_masks'].append(relation_patterns_w_masks)
                feedback_intermediate_results['relation_patterns_w_masks_tagged'].append(relation_patterns_w_masks_tagged)
                feedback_intermediate_results['cluster_assignment'].append(cluster_assignment)
                feedback_intermediate_results['clustered_data_indices_sorted'].append(clustered_data_indices_sorted)

                    
                representative_relation_patterns_examples.append(rel_representative_relation_patterns_examples)
                

            if not os.path.exists(representative_relation_patterns_examples_ckpt):
                torch.save(representative_relation_patterns_examples, representative_relation_patterns_examples_ckpt)

                torch.save(feedback_intermediate_results, os.path.join(llm_ckpt_folder_step1, 'feedback_intermediate_results.pt'))


            rels_pos_examples_followup = self.LLM_followup_pos_generation(rel_set=rel_set, representative_relation_patterns_examples=representative_relation_patterns_examples, init_llm_prompts=init_llm_prompts, num_followup_pos_examples=num_follow_pos_examples, llm_ckpt_folder=llm_ckpt_folder_step1)
            torch.save(rels_pos_examples_followup, rels_pos_examples_followup_ckpt)

        return None
    



###### run_snowball_iterative ######
    def _snowball_iterative_model_train_inference_unlabel_inference(self, rel_id, rel, rel_pos_examples, ckpt_sub_folder, pos_neg_suffix_to_read, pos_neg_suffix_to_save, hist_evaluation_results, prev_rel_hist_evaluation_results, prev_rel_unlabeled_inference_results, rel_def_prompt, rel_def_paraphrased_prompts, prev_chosen_model_state_dict_strategy, test_div, eval_locIds, run_snowball, deterministic, neg_sample_strategy, unlabeled_corpus_div, neg_pos_ratio, writer, print_hierarchy, prev_ckpt_sub_folder=None, random_seed_multiplier=1):
        """
        Sample negative sampels according to the strategy. Read previous pos and neg examples. Log current pos neg examples. Train and evaluate the model and conduct inference on unlabeled corpus.

        Args:
            prev_chosen_model_state_dict_strategy: if set to None, will not load pretrained model state dict before training. If set to 'elbow', will load elbow epoch's pretrained model state dict. If set to 'last_save_epoch', will load the epoch's model state dict referenced by last int in self.args.save_epochs for training. If set to any int, will load the epoch's model state dict referenced by that int (note need to make sure corresponding epoch's model state dict file exists.)


        Assume: self.args.save_epochs stays the same in the 0-th step of snowball and the iterative snowball
        """
        # gather previously pos and neg examples to load
        prev_rel_pos_examples, prev_rel_neg_examples = [], []
        if isinstance(pos_neg_suffix_to_read, list):
            for pos_neg_suffix_to_read_i in pos_neg_suffix_to_read:
                prev_rel_pos_examples.extend(prev_rel_hist_evaluation_results[f'pos_examples{pos_neg_suffix_to_read_i}'])
                prev_rel_neg_examples.extend(prev_rel_hist_evaluation_results[f'neg_examples{pos_neg_suffix_to_read_i}'])
        else:
            prev_rel_pos_examples = prev_rel_hist_evaluation_results[f'pos_examples{pos_neg_suffix_to_read}']
            prev_rel_neg_examples = prev_rel_hist_evaluation_results[f'neg_examples{pos_neg_suffix_to_read}']



        # sample from unlabeled corpus for negative samples
        if deterministic: random.seed(min(self.args.seed, 1) * (rel_id + 1) * random_seed_multiplier)
        if neg_sample_strategy == 'random_all':
            rel_neg_examples_indices =  random.sample(list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data']))), math.ceil(len(rel_pos_examples) * neg_pos_ratio))
        elif neg_sample_strategy == 'random_prior':
            rel_neg_examples_indices_candidates = np.where(prev_rel_unlabeled_inference_results['pred_probs'][:, 1] < self.args.negative_sampling_upper_bound)[0].tolist()
            if len(rel_neg_examples_indices_candidates) < math.ceil(len(rel_pos_examples) * neg_pos_ratio): 
                print(f"{print_hierarchy}len(rel_neg_examples_indices_candidates) ({len(rel_neg_examples_indices_candidates)}) < math.ceil(len(rel_pos_examples) * neg_pos_ratio) ({math.ceil(len(rel_pos_examples) * neg_pos_ratio)}), consider choosing another threshold. For this round, we still sample from all unlabeled corpus.")
                rel_neg_examples_indices_candidates = list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])))
            rel_neg_examples_indices = random.sample(rel_neg_examples_indices_candidates, math.ceil(len(rel_pos_examples) * neg_pos_ratio))
        rel_neg_examples = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_neg_examples_indices)


        # add current constructed pos/neg examples and all previous pos/neg examples to rel evaluation results
        rel_extra_info_to_save = {
            f'pos_examples{pos_neg_suffix_to_save}': rel_pos_examples,
            f'neg_examples{pos_neg_suffix_to_save}': rel_neg_examples,
        }
        for eval_results_key in prev_rel_hist_evaluation_results:
            if (eval_results_key.startswith('pos_examples') and eval_results_key != 'pos_examples') or (eval_results_key.startswith('neg_examples') and eval_results_key != 'neg_examples'):
                rel_extra_info_to_save[eval_results_key] = prev_rel_hist_evaluation_results[eval_results_key]



        # to remove duplicates and invalid pos/neg examples before training
        combined_rel_pos_examples = prev_rel_pos_examples + rel_pos_examples
        combined_rel_pos_examples = remove_duplicated_invalid_examples(examples=combined_rel_pos_examples)
        combined_rel_neg_examples = prev_rel_neg_examples + rel_neg_examples
        combined_rel_neg_examples = remove_duplicated_invalid_examples(examples=combined_rel_neg_examples)
        rel_pos_neg_examples = combined_rel_pos_examples + combined_rel_neg_examples
        rel_pos_neg_assigned_labels = [torch.tensor([1], dtype=torch.long)] * len(combined_rel_pos_examples) + [torch.tensor([0], dtype=torch.long)] * len(combined_rel_neg_examples)


        rel_NLI_model = NLIBasedSimilarityModel(args=self.args, tokenizer=self.dataloader.tokenizer, target_relation=rel, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts)
        if prev_chosen_model_state_dict_strategy is not None: 
            if prev_chosen_model_state_dict_strategy == 'elbow':
                prev_chosen_model_state_dict_epoch = prev_rel_hist_evaluation_results['elbow_epoch']
            elif prev_chosen_model_state_dict_strategy == 'last_save_epoch':
                prev_chosen_model_state_dict_epoch = self.args.save_epochs[-1]
            elif isinstance(prev_chosen_model_state_dict_strategy, int):
                prev_chosen_model_state_dict_epoch = prev_chosen_model_state_dict_strategy

            prev_chosen_model_state_dict_path = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, prev_ckpt_sub_folder, f'{rel}/epoch_model_ckpt/', f"model_ckpt_epoch_{prev_chosen_model_state_dict_epoch}.pt")

            assert os.path.exists(prev_chosen_model_state_dict_path), f"{prev_chosen_model_state_dict_path} does not exist! Variable prev_chosen_model_state_dict_strategy ({prev_chosen_model_state_dict_strategy}) needs to be revised!"

            update_model_state_dict(model=rel_NLI_model, target_state_dict=torch.load(prev_chosen_model_state_dict_path, map_location='cpu'))


        test_div2local_indices = {
            test_div: eval_locIds,
        }

        rel_hist_evaluation_results, ckpt_exists, rel_NLI_model = self.NLIBased_Optim_w_PosNeg_rel(rel_NLI_model=rel_NLI_model, target_rel=rel, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, div2local_indices=None, div2assigned_labels=None, test_div2local_indices=test_div2local_indices, ckpt_sub_folder=ckpt_sub_folder, train_examples=rel_pos_neg_examples, train_assigned_labels=rel_pos_neg_assigned_labels, SummaryWriter=writer, extra_info_to_save=rel_extra_info_to_save)


        if ckpt_exists:
            print(f"{print_hierarchy}===Ckpt found for relation: {rel}. Skipped its optimization round. Use the ckpt as an approximation instead.===")

        hist_evaluation_results['rel_hist_evaluation_results'].append(rel_hist_evaluation_results)

        if run_snowball == False: 
            if os.path.exists(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')):
                print(f"{print_hierarchy}===run_snowball is False, but found cached unlabeled corpus inference results. Load cached unlabeled corpus inference results to hist_evaluation_results.===")
                hist_evaluation_results['rel_unlabeled_inference_results'].append(torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')))
            else:
                print(f"{print_hierarchy}===run_snowball is False, didn't found cached unlabeled corpus inference results. Appending an empty python dictionary to hist_evaluation_results.===")
                hist_evaluation_results['rel_unlabeled_inference_results'].append({})

            return None
        

        if ckpt_exists and os.path.exists(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')):
            print(f"{print_hierarchy}===Cached unlabeled corpus inference results found for relation: {rel}. Skipping its optimization round and its unlabeled corpus inference round. Use cached unlabeled corpus inference instead.===")
            hist_evaluation_results['rel_unlabeled_inference_results'].append(torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')))

            return None
        

        gc.collect()
        torch.cuda.empty_cache() 

        print(f'{print_hierarchy}Before inference on unlabeled corpus, model is on device=', next(rel_NLI_model.parameters()).device)

        chosen_model_state_dict_path = rel_hist_evaluation_results['chosen_rel_NLI_model_state_dict_path']

        # get all the unlabeled corpus examples and conduct inference over them
        rel_unlabeled_indices = list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])))
        rel_unlabeled_examples = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_unlabeled_indices)

        free_port = find_free_port(start_port=self.args.dist_port, end_port=self.args.dist_port+200)
        print(f"{print_hierarchy}Found free port at {free_port}, switching from {self.args.dist_port}")
        self.args.dist_port = free_port

        manager = mp.Manager()
        shared_results = manager.list()
        mp.spawn(NLIBased_Inference, args=(self.args.num_gpus, self.args, chosen_model_state_dict_path, self.dataloader, rel_unlabeled_examples, rel, rel_def_prompt, rel_def_paraphrased_prompts, shared_results, ckpt_sub_folder, 0.5), nprocs=self.args.num_gpus)
        rel_unlabeled_inference_results = list(shared_results)[0]


        hist_evaluation_results['rel_unlabeled_inference_results'].append(rel_unlabeled_inference_results)

        return None




    def _snowball_iterative_model_train_inference(self, rel_id, rel, rel_pos_examples, ckpt_sub_folder, pos_neg_suffix_to_read, pos_neg_suffix_to_save, hist_evaluation_results, prev_rel_hist_evaluation_results, prev_rel_unlabeled_inference_results, rel_def_prompt, rel_def_paraphrased_prompts, prev_chosen_model_state_dict_strategy, test_div, eval_locIds, run_snowball, deterministic, neg_sample_strategy, unlabeled_corpus_div, neg_pos_ratio, writer, print_hierarchy, prev_ckpt_sub_folder=None, random_seed_multiplier=1, rel_neg_examples_followup=None, rel_dev_pos_examples=None):
        """
        Sample negative sampels according to the strategy. Read previous pos and neg examples. Log current pos neg examples. Train and evaluate the model.

        Args:
            prev_chosen_model_state_dict_strategy: if set to None, will not load pretrained model state dict before training. If set to 'elbow', will load elbow epoch's pretrained model state dict. If set to 'last_save_epoch', will load the epoch's model state dict referenced by last int in self.args.save_epochs for training. If set to any int, will load the epoch's model state dict referenced by that int (note need to make sure corresponding epoch's model state dict file exists.)


        Assume: self.args.save_epochs stays the same in the 0-th step of snowball and the iterative snowball
        """
        # gather previously pos and neg examples to load
        prev_rel_pos_examples, prev_rel_neg_examples = [], []
        prev_rel_dev_pos_examples, prev_rel_dev_neg_examples = [], []
        if isinstance(pos_neg_suffix_to_read, list):
            for pos_neg_suffix_to_read_i in pos_neg_suffix_to_read:
                prev_rel_pos_examples.extend(prev_rel_hist_evaluation_results[f'pos_examples{pos_neg_suffix_to_read_i}'])
                prev_rel_neg_examples.extend(prev_rel_hist_evaluation_results[f'neg_examples{pos_neg_suffix_to_read_i}'])
                
                prev_rel_dev_pos_examples.extend(prev_rel_hist_evaluation_results[f'pos_dev_examples{pos_neg_suffix_to_read_i}'])
                prev_rel_dev_neg_examples.extend(prev_rel_hist_evaluation_results[f'neg_dev_examples{pos_neg_suffix_to_read_i}'])
        else:
            prev_rel_pos_examples = prev_rel_hist_evaluation_results[f'pos_examples{pos_neg_suffix_to_read}']
            prev_rel_neg_examples = prev_rel_hist_evaluation_results[f'neg_examples{pos_neg_suffix_to_read}']
            
            prev_rel_dev_pos_examples = prev_rel_hist_evaluation_results[f'pos_dev_examples{pos_neg_suffix_to_read_i}']
            prev_rel_dev_neg_examples = prev_rel_hist_evaluation_results[f'neg_dev_examples{pos_neg_suffix_to_read_i}']

        buffer_factor = 10
        # sample from unlabeled corpus for negative samples
        if deterministic: random.seed(min(self.args.seed, 1) * (rel_id + 1) * random_seed_multiplier)
        if rel_neg_examples_followup is not None:
            rel_neg_examples = deepcopy(rel_neg_examples_followup)
            if self.args.num_follow_neg_examples > self.args.num_follow_neg_examples_to_generate:
                num_follow_neg_examples_to_sample = self.args.num_follow_neg_examples - self.args.num_follow_neg_examples_to_generate
                rel_follow_neg_indices = random.sample(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])), num_follow_neg_examples_to_sample * buffer_factor)
                rel_neg_examples_buffered = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_follow_neg_indices)
                rel_neg_examples.extend(select_valid_distinct_examples(ref_examples=prev_rel_neg_examples + rel_neg_examples, buffered_examples=rel_neg_examples_buffered, target_num_samples=num_follow_neg_examples_to_sample))
                print(f"LLM generated followup negative examples (first {len(rel_neg_examples) - num_follow_neg_examples_to_sample}) and randomly sampled initial negative examples (last {num_follow_neg_examples_to_sample}): ")
                self.dataloader.print_div_raw_data_w_indices(div_list=None, local_indices=None, indent_char='|\t', example_list=rel_neg_examples)
        else:
            if neg_sample_strategy == 'random_all':
                rel_neg_examples_indices =  random.sample(list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data']))), self.args.num_follow_neg_examples * buffer_factor)
                
            elif neg_sample_strategy == 'random_prior':
                rel_neg_examples_indices_candidates = np.where(prev_rel_unlabeled_inference_results['pred_probs'][:, 1] < self.args.negative_sampling_upper_bound)[0].tolist()
                if len(rel_neg_examples_indices_candidates) < self.args.num_follow_neg_examples * buffer_factor:
                    print(f"{print_hierarchy}len(rel_neg_examples_indices_candidates) ({len(rel_neg_examples_indices_candidates)}) < self.args.num_follow_neg_examples * buffer_factor ({self.args.num_follow_neg_examples * buffer_factor}), consider choosing another threshold. For this round, we still sample from all unlabeled samples with scores below the threshold.")
                    rel_neg_examples_indices = rel_neg_examples_indices_candidates
                else:
                    rel_neg_examples_indices = random.sample(rel_neg_examples_indices_candidates, self.args.num_follow_neg_examples * buffer_factor)
            rel_neg_examples_buffered = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_neg_examples_indices)

            rel_neg_examples = select_valid_distinct_examples(ref_examples=prev_rel_neg_examples, buffered_examples=rel_neg_examples_buffered, target_num_samples=self.args.num_follow_neg_examples)
        
        
        rel_follow_neg_indices = random.sample(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])), self.args.num_follow_neg_examples * buffer_factor)
        rel_dev_neg_examples_buffered = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_follow_neg_indices)
        rel_dev_neg_examples = select_valid_distinct_examples(ref_examples=prev_rel_neg_examples + rel_neg_examples + prev_rel_dev_neg_examples, buffered_examples=rel_dev_neg_examples_buffered, target_num_samples=self.args.num_follow_neg_examples)
        
        
        # add current constructed pos/neg examples and all previous pos/neg examples to rel evaluation results
        rel_extra_info_to_save = {
            f'pos_examples{pos_neg_suffix_to_save}': rel_pos_examples,
            f'neg_examples{pos_neg_suffix_to_save}': rel_neg_examples,
            f'pos_dev_examples{pos_neg_suffix_to_save}': rel_dev_pos_examples,
            f'neg_dev_examples{pos_neg_suffix_to_save}': rel_dev_neg_examples,
        }
        for eval_results_key in prev_rel_hist_evaluation_results:
            if (eval_results_key.startswith('pos_examples') and eval_results_key != 'pos_examples') or (eval_results_key.startswith('neg_examples') and eval_results_key != 'neg_examples'):
                rel_extra_info_to_save[eval_results_key] = prev_rel_hist_evaluation_results[eval_results_key]

            if (eval_results_key.startswith('pos_dev_examples') and eval_results_key != 'pos_dev_examples') or (eval_results_key.startswith('neg_dev_examples') and eval_results_key != 'neg_dev_examples'):
                rel_extra_info_to_save[eval_results_key] = prev_rel_hist_evaluation_results[eval_results_key]


        # to remove duplicates and invalid pos/neg examples before training
        combined_rel_pos_examples = prev_rel_pos_examples + rel_pos_examples
        combined_rel_pos_examples = remove_duplicated_invalid_examples(examples=combined_rel_pos_examples)
        combined_rel_neg_examples = prev_rel_neg_examples + rel_neg_examples
        combined_rel_neg_examples = remove_duplicated_invalid_examples(examples=combined_rel_neg_examples)
        rel_pos_neg_examples = combined_rel_pos_examples + combined_rel_neg_examples
        print(f"{print_hierarchy}\t Training RE model with {len(combined_rel_pos_examples)} positive examples and {len(combined_rel_neg_examples)} negative examples in total.")
        rel_pos_neg_assigned_labels = [torch.tensor([1], dtype=torch.long)] * len(combined_rel_pos_examples) + [torch.tensor([0], dtype=torch.long)] * len(combined_rel_neg_examples)


        prev_rel_dev_pos_neg_examples = prev_rel_dev_pos_examples + prev_rel_dev_neg_examples
        prev_rel_dev_pos_neg_assigned_labels = [torch.tensor([1], dtype=torch.long)] * len(prev_rel_dev_pos_examples) + [torch.tensor([0], dtype=torch.long)] * len(prev_rel_dev_neg_examples)
        curr_rel_dev_pos_neg_examples = rel_dev_pos_examples + rel_dev_neg_examples
        curr_rel_dev_pos_neg_assigned_labels = [torch.tensor([1], dtype=torch.long)] * len(rel_dev_pos_examples) + [torch.tensor([0], dtype=torch.long)] * len(rel_dev_neg_examples)
        combined_rel_dev_pos_examples = prev_rel_dev_pos_examples + rel_dev_pos_examples
        combined_rel_dev_pos_examples = remove_duplicated_invalid_examples(examples=combined_rel_dev_pos_examples)
        combined_rel_dev_neg_examples = prev_rel_dev_neg_examples + rel_dev_neg_examples
        combined_rel_dev_neg_examples = remove_duplicated_invalid_examples(examples=combined_rel_dev_neg_examples)
        rel_dev_pos_neg_examples = combined_rel_dev_pos_examples + combined_rel_dev_neg_examples
        rel_dev_pos_neg_assigned_labels = [torch.tensor([1], dtype=torch.long)] * len(combined_rel_dev_pos_examples) + [torch.tensor([0], dtype=torch.long)] * len(combined_rel_dev_neg_examples)



        rel_NLI_model = NLIBasedSimilarityModel(args=self.args, tokenizer=self.dataloader.tokenizer, target_relation=rel, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts)
        if prev_chosen_model_state_dict_strategy is not None: 
            if prev_chosen_model_state_dict_strategy == 'elbow':
                prev_chosen_model_state_dict_epoch = prev_rel_hist_evaluation_results['elbow_epoch']
            elif prev_chosen_model_state_dict_strategy == 'last_save_epoch':
                prev_chosen_model_state_dict_epoch = self.args.save_epochs[-1]
            elif isinstance(prev_chosen_model_state_dict_strategy, int):
                prev_chosen_model_state_dict_epoch = prev_chosen_model_state_dict_strategy

            prev_chosen_model_state_dict_path = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, prev_ckpt_sub_folder, f'{rel}/epoch_model_ckpt/', f"model_ckpt_epoch_{prev_chosen_model_state_dict_epoch}.pt")

            assert os.path.exists(prev_chosen_model_state_dict_path), f"{prev_chosen_model_state_dict_path} does not exist! Variable prev_chosen_model_state_dict_strategy ({prev_chosen_model_state_dict_strategy}) needs to be revised!"

            update_model_state_dict(model=rel_NLI_model, target_state_dict=torch.load(prev_chosen_model_state_dict_path, map_location='cpu'))


        test_div2local_indices = {
            test_div: eval_locIds,
        }

        rel_hist_evaluation_results, ckpt_exists, rel_NLI_model = self.NLIBased_Optim_w_PosNeg_rel(rel_NLI_model=rel_NLI_model, target_rel=rel, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, div2local_indices=None, div2assigned_labels=None, test_div2local_indices=test_div2local_indices, ckpt_sub_folder=ckpt_sub_folder, train_examples=rel_pos_neg_examples, train_assigned_labels=rel_pos_neg_assigned_labels, SummaryWriter=writer, extra_info_to_save=rel_extra_info_to_save, dev_examples=prev_rel_dev_pos_neg_examples, dev_assigned_labels=prev_rel_dev_pos_neg_assigned_labels, dev_examples_1=curr_rel_dev_pos_neg_examples, dev_assigned_labels_1=curr_rel_dev_pos_neg_assigned_labels, dev_examples_2=rel_dev_pos_neg_examples, dev_assigned_labels_2=rel_dev_pos_neg_assigned_labels,)


        if ckpt_exists:
            print(f"{print_hierarchy}===Ckpt found for relation: {rel}. Skipped its optimization round. Use the ckpt as an approximation instead.===")

        hist_evaluation_results['rel_hist_evaluation_results'].append(rel_hist_evaluation_results)
        hist_evaluation_results['train_inference_ckpt_exists'].append(ckpt_exists)

        return None



    def _snowball_iterative_model_unlabel_inference(self, rel, ckpt_sub_folder, hist_evaluation_results, rel_def_prompt, rel_def_paraphrased_prompts, run_snowball, rel_hist_evaluation_results, unlabeled_corpus_div, print_hierarchy, ckpt_exists=False):
        """
        Conduct inference on unlabeled corpus.

        Args:
            prev_chosen_model_state_dict_strategy: if set to None, will not load pretrained model state dict before training. If set to 'elbow', will load elbow epoch's pretrained model state dict. If set to 'last_save_epoch', will load the epoch's model state dict referenced by last int in self.args.save_epochs for training. If set to any int, will load the epoch's model state dict referenced by that int (note need to make sure corresponding epoch's model state dict file exists.)


        Assume: self.args.save_epochs stays the same in the 0-th step of snowball and the iterative snowball
        """
        if run_snowball == False: 
            if os.path.exists(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')):
                print(f"{print_hierarchy}===run_snowball is False, but found cached unlabeled corpus inference results. Load cached unlabeled corpus inference results to hist_evaluation_results.===")
                hist_evaluation_results['rel_unlabeled_inference_results'].append(torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')))
            else:
                print(f"{print_hierarchy}===run_snowball is False, didn't found cached unlabeled corpus inference results. Appending an empty python dictionary to hist_evaluation_results.===")
                hist_evaluation_results['rel_unlabeled_inference_results'].append({})

            return None
        

        if ckpt_exists and os.path.exists(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')):
            print(f"{print_hierarchy}===run_snowball is True. Cached unlabeled corpus inference results found for relation: {rel}. Skipping its optimization round and its unlabeled corpus inference round. Use cached unlabeled corpus inference instead.===")
            hist_evaluation_results['rel_unlabeled_inference_results'].append(torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{rel}_unlabeled_inference.pt')))

            return None
        

        gc.collect()
        torch.cuda.empty_cache() 


        rel_NLI_model = NLIBasedSimilarityModel(args=self.args, tokenizer=self.dataloader.tokenizer, target_relation=rel, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts)
        print(f'{print_hierarchy}Before inference on unlabeled corpus, model is on device=', next(rel_NLI_model.parameters()).device)


        chosen_model_state_dict_path = rel_hist_evaluation_results['dev_chosen_rel_NLI_model_state_dict_path']
        
        # get all the unlabeled corpus examples and conduct inference over them
        rel_unlabeled_indices = list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])))
        rel_unlabeled_examples = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=rel_unlabeled_indices)

        free_port = find_free_port(start_port=self.args.dist_port, end_port=self.args.dist_port+200)
        print(f"{print_hierarchy}Found free port at {free_port}, switching from {self.args.dist_port}")
        self.args.dist_port = free_port

        manager = mp.Manager()
        shared_results = manager.list()
        mp.spawn(NLIBased_Inference, args=(self.args.num_gpus, self.args, chosen_model_state_dict_path, self.dataloader, rel_unlabeled_examples, rel, rel_def_prompt, rel_def_paraphrased_prompts, shared_results, ckpt_sub_folder, 0.5), nprocs=self.args.num_gpus)
        rel_unlabeled_inference_results = list(shared_results)[0]


        hist_evaluation_results['rel_unlabeled_inference_results'].append(rel_unlabeled_inference_results)

        return None


    def _run_snowball_iterative_selftrain(self, snowball_iter_id, pos_neg_suffix_to_read, pos_neg_suffix_to_save, rel_set, eval_locIds, max_paraphrased_prompts, ckpt_sub_folder, prev_hist_evaluation_results, new_threshold=0.75, top_N=2000, neg_sample_strategy='random_all', neg_pos_ratio=2, deterministic=True, prev_chosen_model_state_dict_strategy=None, run_snowball=True, prev_ckpt_sub_folder=None, random_seed_multiplier=1):
        """
        
        Args:
            pos_neg_suffix_to_read: suffix (could be a str or a list of str) to be appended for reading previously cached positive and negative examples. For instance, could be ['_0', '_1'] or just '_0'
            pos_neg_suffix_to_save: suffix (a str) to be appended to caching result dict's keys for newly mined positive and negative samples. Note that in the snowball step 0, we have suffix as '_0'
            ckpt_sub_folder: "accumulative" folder for saving this function's cache and results. Will be used as: self.args.dataset_dir/self.args.cache_sub_dir/ckpt_sub_folder/...
            top_N: cannot be a reused / to-be-reused variable, try to deepcopy when passing values to this argument
            prev_chosen_model_state_dict_strategy: if set to None, will not load any pretrained model state dict; if set to 


        Functionalities to implement: 
            - ckpt_sub_folder: this can be a sub path of multiple hierarchy which enables iterative snowball
            - choose which epoch in epochs_to_save to load for further training (None represents do not load pretrained RE model ckpt). This is decided by argument prev_chosen_model_state_dict_strategy and should be defined by upstream functions. 
            - choose which previous positive and negative samples to consider with new pos and neg examples to train the model. This is realized by argument pos_neg_suffix_to_read and pos_neg_suffix_to_save which should be decided by upsteam functions.
            - automatically load the cached results if exist and skip to save time

            
        Assume: top_N * neg_pos_ratio <= |unlabeled corpus|
        """
        
        print_hierarchy = f"[Iter-{snowball_iter_id}/SelfTrain]: Set threshold={new_threshold}, top_N={top_N}, neg_pos_ratio={neg_pos_ratio}"

        prev_unlabeled_inference_results = prev_hist_evaluation_results['rel_unlabeled_inference_results']
        test_div = prev_hist_evaluation_results['test_div']
        unlabeled_corpus_div = prev_hist_evaluation_results['unlabeled_corpus_div']
        assert rel_set == prev_hist_evaluation_results['rel_set']

        os.makedirs(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, 'tensorboard/'))
        

        # evaluation results of this function
        hist_evaluation_results = {
            'test_div': test_div,
            'unlabeled_corpus_div': unlabeled_corpus_div,
            'args': self.args,
            'rel_set': rel_set,
            'rel': [],
            'eval_locIds': eval_locIds,
            'rel_hist_evaluation_results': [],
            'rel_unlabeled_inference_results': [],
            'train_inference_ckpt_exists': [],
        }


        for rel_id, rel in enumerate(rel_set):
            print(f"{print_hierarchy}===Self training on relation: {rel} ({self.dataloader.rel_info[rel]['relation_name']})===")            
            hist_evaluation_results['rel'].append(rel)

            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
            print(f"{print_hierarchy}|\tRelation definition prompt for {rel}: {rel_def_prompt}")
            for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                print(f"{print_hierarchy}|\t\tParaphrased prompt {i} for {rel}: {rel_def_paraphrased_prompt}")


            prev_rel_hist_evaluation_results = prev_hist_evaluation_results['rel_hist_evaluation_results'][rel_id]
            prev_rel_unlabeled_inference_results = prev_unlabeled_inference_results[rel_id]
            assert rel == prev_rel_hist_evaluation_results['rel'] and prev_rel_unlabeled_inference_results['inference_relative_ids'].tolist() == list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])))


            prev_rel_unlabeled_pred_probs = prev_rel_unlabeled_inference_results['pred_probs']
            prev_rel_unlabeled_inference_relative_ids = prev_rel_unlabeled_inference_results['inference_relative_ids']
            count_above_threshold = np.sum(prev_rel_unlabeled_pred_probs[:, 1] >= new_threshold)
            if count_above_threshold < top_N: 
                print(f"{print_hierarchy}Number of pred probs >= new_threshold ({count_above_threshold}) is smaller than top_N. Adjusting top_N to this number.")
                top_N = count_above_threshold
            sorted_indices = np.argsort(prev_rel_unlabeled_pred_probs[:, 1])[::-1] # numpy array of (|unlabeled corpus|, ), indicating the indices from largest pos pred prob to least
            prev_rel_unlabeled_inference_relative_ids_sorted = prev_rel_unlabeled_inference_relative_ids[sorted_indices] # numpy array of (|unlabeled corpus|, ), indicating the indices to the unlabeled corpus samples sorted by descending pos pred prob.
            prev_top_N_unlabeled_indices = prev_rel_unlabeled_inference_relative_ids_sorted[:top_N]
            
            rel_pos_examples = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=prev_top_N_unlabeled_indices)
            rel_pos_examples = remove_duplicated_invalid_examples(examples=rel_pos_examples)


            self._snowball_iterative_model_train_inference(rel_id=rel_id, rel=rel, rel_pos_examples=rel_pos_examples, ckpt_sub_folder=ckpt_sub_folder, pos_neg_suffix_to_read=pos_neg_suffix_to_read, pos_neg_suffix_to_save=pos_neg_suffix_to_save, hist_evaluation_results=hist_evaluation_results, prev_rel_hist_evaluation_results=prev_rel_hist_evaluation_results, prev_rel_unlabeled_inference_results=prev_rel_unlabeled_inference_results, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, prev_chosen_model_state_dict_strategy=prev_chosen_model_state_dict_strategy, test_div=test_div, eval_locIds=eval_locIds, run_snowball=run_snowball, deterministic=deterministic, neg_sample_strategy=neg_sample_strategy, unlabeled_corpus_div=unlabeled_corpus_div, neg_pos_ratio=neg_pos_ratio, writer=writer, print_hierarchy=print_hierarchy, prev_ckpt_sub_folder=prev_ckpt_sub_folder, random_seed_multiplier=random_seed_multiplier)


        for epoch_idx in self.args.save_epochs:
            print(f"{print_hierarchy}Results (at epoch={epoch_idx}) averaged over all relations: P: {np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")

        print(f"{print_hierarchy}Chosen results averaged over all relations: P: {np.mean([i['chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")


        for epoch_idx in range(self.args.num_train_epochs):
            avg_precision = np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged f1", avg_f1, epoch_idx)


        gc.collect()
        torch.cuda.empty_cache() 

        for rel_id, rel in enumerate(rel_set):
            rel_ckpt_exists = hist_evaluation_results['train_inference_ckpt_exists'][rel_id]
            rel_hist_evaluation_results = hist_evaluation_results['rel_hist_evaluation_results'][rel_id]

            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]

            self._snowball_iterative_model_unlabel_inference(rel=rel, ckpt_sub_folder=ckpt_sub_folder, hist_evaluation_results=hist_evaluation_results, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, run_snowball=run_snowball, rel_hist_evaluation_results=rel_hist_evaluation_results, unlabeled_corpus_div=unlabeled_corpus_div, print_hierarchy=print_hierarchy, ckpt_exists=rel_ckpt_exists)


        self.save_model(
            os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_all.pt'),
            hist_evaluation_results,
        )


        return hist_evaluation_results



    def _snowball_iterative_get_rels_pos_examples_followup(self, rels_pos_examples_followup_ckpt, prev_llm_ckpt_folder, llm_ckpt_folder, rel_set, max_paraphrased_prompts, prev_hist_evaluation_results, num_follow_pos_examples, new_threshold=0.85, top_N=50000, print_hierarchy="", cluster_patterns=False, feedback_w_scores=False, sliding_window_size=10, enable_sliding_window=True, rels_dev_pos_examples_followup_ckpt=None):
        """
        
        Args:
            llm_ckpt_folder: [assume this folder is already created]

        """
        

        if not os.path.exists(rels_pos_examples_followup_ckpt):
            if os.path.exists(os.path.join(prev_llm_ckpt_folder, 'gpt4_chat_pos.jsonl')): init_llm_prompts = load_jsonl(filename=os.path.join(prev_llm_ckpt_folder, 'gpt4_chat_pos.jsonl'))
            else: init_llm_prompts = load_jsonl(filename=os.path.join(prev_llm_ckpt_folder, 'gpt4_chat.jsonl'))
                
            prev_unlabeled_inference_results = prev_hist_evaluation_results['rel_unlabeled_inference_results']
            unlabeled_corpus_div = prev_hist_evaluation_results['unlabeled_corpus_div']


            # https://spacy.io/usage/processing-pipelines#_title for more info on spacy nlp.pipe
            spacy.prefer_gpu()
            # nlp = spacy.load("en_core_web_lg")
            nlp = spacy.load("en_core_web_trf")
            n_cores = multiprocessing.cpu_count()


            representative_relation_patterns_examples = []
            representative_relation_patterns_examples_scores = []
            representative_relation_patterns_examples_ckpt = os.path.join(llm_ckpt_folder, 'feedback_examples_pos.pt')
            if os.path.exists(representative_relation_patterns_examples_ckpt): representative_relation_patterns_examples = torch.load(representative_relation_patterns_examples_ckpt)

            if feedback_w_scores:
                representative_relation_patterns_examples_scores_ckpt = os.path.join(llm_ckpt_folder, 'feedback_examples_pos_scores.pt')
                if os.path.exists(representative_relation_patterns_examples_scores_ckpt): representative_relation_patterns_examples_scores = torch.load(representative_relation_patterns_examples_scores_ckpt)
            
            
            feedback_intermediate_results = {
                'cluster_patterns': cluster_patterns,
                'feedback_w_scores': feedback_w_scores,
                'relation_patterns': [],
                'relation_patterns_examples': [],
                'relation_patterns_w_masks': [],
                'relation_patterns_w_masks_tagged': [],
                'cluster_assignment': [],
                'clustered_data_indices_sorted': [],
            }

            for rel_id, rel in enumerate(rel_set):
                if os.path.exists(representative_relation_patterns_examples_ckpt):
                    print(f"{print_hierarchy}\tCached feedback examples file exists. Skipping mining process for {rel} ({self.dataloader.rel_info[rel]['relation_name']})...")
                    print(f"{print_hierarchy}\tFeedback examples: ")
                    self.dataloader.print_div_raw_data_w_indices(example_list=representative_relation_patterns_examples[rel_id], indent_char='|\t')
                    continue
                
                print(f"{print_hierarchy}===Mining relation pattern feedbacks to LLM: {rel} ({self.dataloader.rel_info[rel]['relation_name']})===")
                if cluster_patterns: print(f"{print_hierarchy}|\tPattern Mining Method: pattern extraction and clustering")
                else: print(f"{print_hierarchy}|\tPattern Mining Method: random sampling")
                
                
                rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
                rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
                print(f"{print_hierarchy}|\tRelation definition prompt for {rel}: {rel_def_prompt}")
                for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                    print(f"{print_hierarchy}|\t\tParaphrased prompt {i} for {rel}: {rel_def_paraphrased_prompt}")
                
                
                prev_rel_hist_evaluation_results = prev_hist_evaluation_results['rel_hist_evaluation_results'][rel_id]
                prev_rel_unlabeled_inference_results = prev_unlabeled_inference_results[rel_id]
                assert (rel == prev_rel_hist_evaluation_results['rel']) and (prev_rel_unlabeled_inference_results['inference_relative_ids'].tolist() == list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data']))))

                rel_new_threshold = deepcopy(new_threshold)
                rel_top_N = deepcopy(top_N)
                print(f"{print_hierarchy}\tSetting the rel_new_threshold={rel_new_threshold} and setting rel_top_N={rel_top_N} for selecting positive instances from unlabled corpus")
                prev_rel_unlabeled_pred_probs = prev_rel_unlabeled_inference_results['pred_probs']
                prev_rel_unlabeled_inference_relative_ids = prev_rel_unlabeled_inference_results['inference_relative_ids']
                count_above_threshold = np.sum(prev_rel_unlabeled_pred_probs[:, 1] >= rel_new_threshold)
                if count_above_threshold < rel_top_N:
                    print(f"{print_hierarchy}\tNumber of pred probs >= rel_new_threshold ({count_above_threshold}) is smaller than rel_top_N. Adjusting rel_top_N to this number.")
                    rel_top_N = max(40, count_above_threshold) # changed
                    print("Adjusted rel_top_N to: ", rel_top_N)
                    
                sorted_indices = np.argsort(prev_rel_unlabeled_pred_probs[:, 1])[::-1] # numpy array of (|unlabeled corpus|, ), indicating the indices from largest pos pred prob to least
                prev_rel_unlabeled_inference_relative_ids_sorted = prev_rel_unlabeled_inference_relative_ids[sorted_indices] # numpy array of (|unlabeled corpus|, ), indicating the indices to the unlabeled corpus samples sorted by descending pos pred prob
                prev_top_N_unlabeled_indices = prev_rel_unlabeled_inference_relative_ids_sorted[:rel_top_N]
                prev_rel_top_unlabeled_examples = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=prev_top_N_unlabeled_indices)
                

                if feedback_w_scores:
                    prev_rel_top_unlabeled_examples, prev_rel_top_unlabeled_examples_pos_scores = remove_duplicated_invalid_examples_v1(examples=prev_rel_top_unlabeled_examples, scores=prev_rel_unlabeled_pred_probs[:, 1][sorted_indices[:rel_top_N]])
                else:
                    prev_rel_top_unlabeled_examples = remove_duplicated_invalid_examples(examples=prev_rel_top_unlabeled_examples)

                if cluster_patterns:
                    preprocessed_docs = list(nlp.pipe([" ".join(unlabeled_ex['tokens']) for unlabeled_ex in prev_rel_top_unlabeled_examples], batch_size=2048))

                    h_t_unconnected_count = 0 # counting the number of examples where head ent and tail ent are not in the same dep parsing tree
                    relation_patterns = []
                    relation_patterns_examples = []
                    relation_patterns_w_masks = []
                    relation_patterns_w_masks_tagged = []
                    rel_representative_relation_patterns_examples = []
                    rel_representative_relation_patterns_examples_scores = []
                    relation_patterns_original_relative_indices = []
                    
                    
                    for unlabeled_ex_id, unlabeled_ex in enumerate(tqdm(prev_rel_top_unlabeled_examples, desc='Examples')):
                        doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path = get_shortest_dep_path_v3(word_list=unlabeled_ex['tokens'], ht_word_spans=[[unlabeled_ex['h'][2][0][0], unlabeled_ex['h'][2][0][-1]], [unlabeled_ex['t'][2][0][0], unlabeled_ex['t'][2][0][-1]]], doc_given=preprocessed_docs[unlabeled_ex_id])

                        if h_doc_span_path is None:
                            h_t_unconnected_count += 1
                            continue
                        
                        h_dep_t_string = " ".join(linearize_ent_dep_path(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, add_tags=False, add_h_t=True))
                        h_dep_t_string_w_mask = " ".join(linearize_ent_dep_path_w_mask(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, mask_h_t=False, mask_token='<mask>', tag_h_t=False))
                        h_dep_t_string_w_mask_tagged = " ".join(linearize_ent_dep_path_w_mask(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, mask_h_t=False, mask_token='<mask>', tag_h_t=True))
                        
                        relation_patterns_original_relative_indices.append(unlabeled_ex_id)
                        relation_patterns.append(h_dep_t_string)
                        relation_patterns_examples.append(unlabeled_ex)
                        relation_patterns_w_masks.append(h_dep_t_string_w_mask)
                        relation_patterns_w_masks_tagged.append(h_dep_t_string_w_mask_tagged)

                    
                    relation_patterns_embs = get_SBERT_embedding(sent_list=relation_patterns_w_masks, sbert_model='all-mpnet-base-v2', batch_size=768, normalize_embeddings=True)

                    cluster_centroids, cluster_assignment, clustered_data_indices, clustered_dists, dist = get_KMeans_clusters(data_embeddings=relation_patterns_embs, seed=self.args.seed, n_clusters=10)
                    clustered_data_indices_sorted = []
                    for centroid_idx in range(len(clustered_data_indices)):
                        assert len(clustered_data_indices[centroid_idx]) == len(clustered_dists[centroid_idx])
                        clustered_data_indices_sorted.append([])
                        sorted_indices = np.argsort(clustered_dists[centroid_idx]) # should be reversed for using HDBSCAN
                        for j in sorted_indices: clustered_data_indices_sorted[-1].append(clustered_data_indices[centroid_idx][j])
                    for centroid_idx in range(len(clustered_data_indices_sorted)):
                        rel_representative_relation_patterns_examples.append(relation_patterns_examples[clustered_data_indices_sorted[centroid_idx][0]])
                        
                        if feedback_w_scores:
                            rel_representative_relation_patterns_examples_scores.append(float(prev_rel_top_unlabeled_examples_pos_scores[relation_patterns_original_relative_indices[clustered_data_indices_sorted[centroid_idx][0]]]))
                        
                    for centroid_idx, clustered_data_ids in enumerate(clustered_data_indices_sorted):
                        print(f'{print_hierarchy}\nCluster {centroid_idx} and its top examples')
                        for j in clustered_data_ids[:20]:
                            print("\t\tPattern: " + relation_patterns_w_masks_tagged[j])
                            print("\t\tExample:" + get_so_tagged_example(example=relation_patterns_examples[j], compact=True))
                            print('--' * 30)
                        print("\n")

                    feedback_intermediate_results['relation_patterns'].append(relation_patterns)
                    feedback_intermediate_results['relation_patterns_examples'].append(relation_patterns_examples)
                    feedback_intermediate_results['relation_patterns_w_masks'].append(relation_patterns_w_masks)
                    feedback_intermediate_results['relation_patterns_w_masks_tagged'].append(relation_patterns_w_masks_tagged)
                    feedback_intermediate_results['cluster_assignment'].append(cluster_assignment)
                    feedback_intermediate_results['clustered_data_indices_sorted'].append(clustered_data_indices_sorted)

                else:
                    sample_num = 30
                    if feedback_w_scores:
                        rel_representative_relation_patterns_examples_ids = random.sample(range(prev_rel_top_unlabeled_examples), k=sample_num)
                        rel_representative_relation_patterns_examples = [prev_rel_top_unlabeled_examples[rel_representative_relation_patterns_examples_i] for rel_representative_relation_patterns_examples_i in rel_representative_relation_patterns_examples_ids]
                        rel_representative_relation_patterns_examples_scores = [float(prev_rel_top_unlabeled_examples_pos_scores[rel_representative_relation_patterns_examples_i]) for rel_representative_relation_patterns_examples_i in rel_representative_relation_patterns_examples_ids]
                    else:
                        rel_representative_relation_patterns_examples = random.sample(prev_rel_top_unlabeled_examples, k=sample_num)
                    
                representative_relation_patterns_examples.append(rel_representative_relation_patterns_examples)
                
                if feedback_w_scores:
                    representative_relation_patterns_examples_scores.append(rel_representative_relation_patterns_examples_scores)

            if not os.path.exists(representative_relation_patterns_examples_ckpt):
                torch.save(representative_relation_patterns_examples, representative_relation_patterns_examples_ckpt)
                torch.save(feedback_intermediate_results, os.path.join(llm_ckpt_folder, 'feedback_intermediate_results_pos.pt'))

            if feedback_w_scores and not os.path.exists(representative_relation_patterns_examples_scores_ckpt):
                torch.save(representative_relation_patterns_examples_scores, representative_relation_patterns_examples_scores_ckpt)
            
            dev_init_llm_prompts = deepcopy(init_llm_prompts)
            
            
            rels_pos_examples_followup = self.LLM_followup_pos_generation_parallel(rel_set=rel_set, representative_relation_patterns_examples=representative_relation_patterns_examples, init_llm_prompts=init_llm_prompts, num_followup_pos_examples=num_follow_pos_examples, llm_ckpt_folder=llm_ckpt_folder, feedback_w_scores=feedback_w_scores, representative_relation_patterns_examples_scores=representative_relation_patterns_examples_scores, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window)
            torch.save(rels_pos_examples_followup, rels_pos_examples_followup_ckpt)

            llm_ckpt_dev_folder = os.path.join(llm_ckpt_folder, 'dev')
            os.makedirs(llm_ckpt_dev_folder, exist_ok=True)
            rels_dev_pos_examples_followup = self.LLM_followup_pos_generation_parallel(rel_set=rel_set, representative_relation_patterns_examples=representative_relation_patterns_examples, init_llm_prompts=dev_init_llm_prompts, num_followup_pos_examples=num_follow_pos_examples, llm_ckpt_folder=llm_ckpt_dev_folder, feedback_w_scores=feedback_w_scores, representative_relation_patterns_examples_scores=representative_relation_patterns_examples_scores, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window)
            torch.save(rels_dev_pos_examples_followup, rels_dev_pos_examples_followup_ckpt)
        return None


    def LLM_followup_neg_def_generation_parallel(self, rel_set, rel_def_prompt_list, representative_relation_patterns_examples, neg_def_gen_template_list, num_neg_rels_to_generate_adjusted, LLM_model, max_num_attempts, task_id, llm_ckpt_folder=None, sliding_window_size=10, enable_sliding_window=True, prev_llm_ckpt_folder_neg=None, feedback_w_scores=False, representative_relation_patterns_examples_scores=None):
        if self.args.run_LLM_json_parser_def:
            chat_response_to_def_prompts_parser = lambda response, num_def_prompts: parse_chat_response_to_def_prompts_LLM_json_parser(response=response, num_def_prompts=num_def_prompts, llm_model_ckpt=self.args.llm_model_ckpt_parser)
        else:
            chat_response_to_def_prompts_parser = parse_chat_response_to_def_prompts_v1
        
        input_list = []
        input_list_x_elt = []
        
        if os.path.exists(os.path.join(prev_llm_ckpt_folder_neg, 'neg_def_prompts.pt')):
            prev_neg_def_prompts = torch.load(os.path.join(prev_llm_ckpt_folder_neg, 'neg_def_prompts.pt'))
            

        for rel_id, (rel, rel_def_prompt) in enumerate(zip(rel_set, rel_def_prompt_list)):
            
            rel_representative_relation_patterns_examples = representative_relation_patterns_examples[rel_id]
            if feedback_w_scores: rel_representative_relation_patterns_examples_scores = representative_relation_patterns_examples_scores[rel_id]
            if not enable_sliding_window:
                feedback_examples_to_print = []
                for ex_id, ex in enumerate(rel_representative_relation_patterns_examples):
                    if feedback_w_scores:
                        feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True) + f"(Predicted probability={rel_representative_relation_patterns_examples_scores[ex_id]:.2f})")
                    else:
                        feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True))
                feedback_examples_to_print_string = "\n\n".join(feedback_examples_to_print)
                
                  
            for neg_def_gen_template_id, neg_def_gen_template in enumerate(neg_def_gen_template_list):
                if enable_sliding_window:
                    feedback_examples_to_print = []
                    for ex_id, ex in enumerate(rel_representative_relation_patterns_examples[neg_def_gen_template_id * sliding_window_size: neg_def_gen_template_id * sliding_window_size + sliding_window_size]):
                        if feedback_w_scores:
                            feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True) + f"(Predicted probability={rel_representative_relation_patterns_examples_scores[neg_def_gen_template_id * sliding_window_size: neg_def_gen_template_id * sliding_window_size + sliding_window_size][ex_id]:.2f})")
                        else:
                            feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True))
                    feedback_examples_to_print_string = "\n\n".join(feedback_examples_to_print)

                if os.path.exists(os.path.join(prev_llm_ckpt_folder_neg, 'neg_def_prompts.pt')):
                    prev_neg_rel_def_prompts_to_print = []
                    for prev_neg_rel_def_prompt_id, prev_neg_rel_def_prompt in enumerate(prev_neg_def_prompts[rel_id]):
                        prev_neg_rel_def_prompts_to_print.append(f"{prev_neg_rel_def_prompt_id + 1}. {prev_neg_rel_def_prompt}")
                    prev_neg_rel_def_prompts_to_print_string = "\n\n".join(prev_neg_rel_def_prompts_to_print)
                        
                    neg_def_gen_prompt = neg_def_gen_template.substitute(pos_rel_def_prompt=rel_def_prompt, feedback_examples=feedback_examples_to_print_string, prev_neg_def_prompts=prev_neg_rel_def_prompts_to_print_string, num_neg_rels=num_neg_rels_to_generate_adjusted)
                else:
                    neg_def_gen_prompt = neg_def_gen_template.substitute(pos_rel_def_prompt=rel_def_prompt, feedback_examples=feedback_examples_to_print_string, num_neg_rels=num_neg_rels_to_generate_adjusted)
                
                input_list.append({
                    'task_id': task_id,
                    'messages': [
                        {"role": "user", "content": neg_def_gen_prompt}
                    ]
                })
                input_list_x_elt.append(f"followup_neg_def_{rel_id}-{neg_def_gen_template_id}")
                task_id += 1
                
        output_list = LLM_model.async_calls(input_list=input_list)
        current_best_output_list = deepcopy(output_list)
        
        followup_neg_def_prompts = []
        
        input_list_to_regen = []
        input_list_to_regen_x_elt = []
        
        current_num_attempts = 0
        for input_idx, (input_x_elt, output_response) in enumerate(zip(input_list_x_elt, output_list)):
            rel_id = int(input_x_elt[17:].split('-')[0])
            neg_def_gen_template_id = int(input_x_elt[17:].split('-')[1])
            rel = rel_set[rel_id]


            temp_neg_rel_def_prompts = chat_response_to_def_prompts_parser(response=deepcopy(output_response), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted))
            
            if len(temp_neg_rel_def_prompts) == num_neg_rels_to_generate_adjusted: followup_neg_def_prompts.append(deepcopy(temp_neg_rel_def_prompts))
            else:
                input_list_to_regen.append(deepcopy(input_list[input_idx]))
                input_list_to_regen_x_elt.append(deepcopy(input_x_elt))
                followup_neg_def_prompts.append(None)

        while (None in followup_neg_def_prompts) and current_num_attempts <= max_num_attempts:
            print(f"Current attempt: {current_num_attempts} with #invalid inputs to regenerate={followup_neg_def_prompts.count(None)}")
            assert followup_neg_def_prompts.count(None) == len(input_list_to_regen)
            
            output_list_to_regen = LLM_model.async_calls(input_list=input_list_to_regen)
            input_list_to_regen_temp = []
            input_list_to_regen_x_elt_temp = []
            
            for input_to_regen_idx, (input_x_elt_to_regen, output_response_to_regen) in enumerate(zip(input_list_to_regen_x_elt, output_list_to_regen)):
                rel_id = int(input_x_elt_to_regen[17:].split('-')[0])
                neg_def_gen_template_id = int(input_x_elt_to_regen[17:].split('-')[1])
                rel = rel_set[rel_id]

                absolute_idx = input_list_x_elt.index(input_x_elt_to_regen)
                
                temp_neg_rel_def_prompts = chat_response_to_def_prompts_parser(response=deepcopy(output_response_to_regen), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted))
                
                if len(temp_neg_rel_def_prompts) == num_neg_rels_to_generate_adjusted: 
                    assert followup_neg_def_prompts[absolute_idx] is None
                    followup_neg_def_prompts[absolute_idx] = deepcopy(temp_neg_rel_def_prompts)
                    current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
                else:
                    input_list_to_regen_temp.append(deepcopy(input_list_to_regen[input_to_regen_idx]))
                    input_list_to_regen_x_elt_temp.append(deepcopy(input_x_elt_to_regen))
                    
                    
                    if abs(len(temp_neg_rel_def_prompts) - num_neg_rels_to_generate_adjusted) <= 2 and len(temp_neg_rel_def_prompts) > len(chat_response_to_def_prompts_parser(response=deepcopy(current_best_output_list[absolute_idx]), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted))):
                        current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
            
            input_list_to_regen = deepcopy(input_list_to_regen_temp)
            input_list_to_regen_x_elt = deepcopy(input_list_to_regen_x_elt_temp)
            
            current_num_attempts += 1
        
        followup_neg_def_prompts_output = []
        for neg_def_prompts_idx, (input_x_elt, neg_def_prompt_list) in enumerate(zip(input_list_x_elt, followup_neg_def_prompts)):
            rel_id = int(input_x_elt[17:].split('-')[0])
            neg_def_gen_template_id = int(input_x_elt[17:].split('-')[1])
            rel = rel_set[rel_id]
            
            if neg_def_gen_template_id == 0: followup_neg_def_prompts_output.append([])
            
            if neg_def_prompt_list is not None:
                followup_neg_def_prompts_output[rel_id].extend(neg_def_prompt_list)
            else:
                followup_neg_def_prompts_output[rel_id].extend(chat_response_to_def_prompts_parser(response=deepcopy(current_best_output_list[neg_def_prompts_idx]), num_def_prompts=deepcopy(num_neg_rels_to_generate_adjusted)))
            
            response = current_best_output_list[neg_def_prompts_idx]
            response_dump = response.model_dump()
            response_dump.update({'prompt_input': input_list[neg_def_prompts_idx], 'relation': f"{rel}->!{rel}", 'relative_id': input_x_elt})
            append_to_jsonl(
                data=response_dump,
                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat_neg_def.jsonl'),
            )
            
            
        return followup_neg_def_prompts_output, task_id
        
    def LLM_followup_neg_generation_parallel(self, rel_set, rel_def_prompt_list, representative_relation_patterns_examples, num_followup_neg_examples=10, llm_ckpt_folder=None, sliding_window_size=10, enable_sliding_window=True, prev_llm_ckpt_folder_neg=None, feedback_w_scores=False, representative_relation_patterns_examples_scores=None):
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat_parallel_neg.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model.update_call_attributes(max_tokens=4096, seed=self.args.seed, temperature=self.args.neg_ex_gen_temperature)
        
        LLM_model_def_gen = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat_parallel_neg.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model_def_gen.update_call_attributes(max_tokens=4096, seed=self.args.seed, temperature=self.args.def_gen_temperature)
        
        max_num_attempts = 2

        neg_def_gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\" (in relation examples, <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> .).\nTypical examples predicted as positive by current relation extraction model are:\n\n${feedback_examples}\n\nExisting generated negative relation definitions are:\n\n${prev_neg_def_prompts}\n\nBased on the positive relation definition, the typical predicted examples and the existing generated negative relation definitions, generate ${num_neg_rels} additional negative binary relation definitions (numbered from 1 to ${num_neg_rels}) in the same format as the above positive relation definition (including entity placeholders and entity type constraints). Other requirements are:\n\n1. Identify false positive predictions from the typical predicted examples and your generated negative relations should teach model to mitigate such bias.\n2. After addressing the previous requirement or if there is no false positive prediction, consider generating near-miss negative relations.\n3. Your generated negative relation definitions should not be the same as existing negative relation definitions."),
        ]
        neg_def_gen_template_list_noIniNeg = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by: \"${pos_rel_def_prompt}\". In relation examples or relation instances, <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> .\nTypical examples predicted as positive by current relation extraction model are:\n\n${feedback_examples}\n\nBased on the positive relation definition and the typical predicted examples, generate ${num_neg_rels} negative binary relation definitions (numbered from 1 to ${num_neg_rels}) in the same format as the above positive relation definition (including entity placeholders and entity type constraints). Other requirements are:\n\n1. Identify false positive predictions from the typical predicted examples and your generated negative relations should teach model to mitigate such bias.\n2. After addressing the previous requirement or if there is no false positive prediction, consider generating near-miss negative relations."),
        ]
        
        gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by: \"${rel_def_prompt}\". Under sentence-level relation extraction setting, generate ${num_examples} examples (numbered from 1 to ${num_examples}) expressing the same relation, where <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> . Other content requirements:\n\n1. Do not overfit the pattern of definition. Try as many different relation patterns or relation expressions as possible.\n2. Generate rich and informative related contexts before and after each entity."),
        ]
        
        if self.args.run_LLM_json_parser_ex:
            chat_response_to_examples_parser = lambda response, relation, num_examples: parse_chat_response_to_examples_LLM_json_parser(response=response, relation=relation, num_examples=num_examples, llm_model_ckpt=self.args.llm_model_ckpt_parser)
        else:
            chat_response_to_examples_parser = parse_chat_response_to_examples
        
        
        num_followup_neg_rels_to_generate_adjusted = math.ceil(self.args.num_follow_neg_rels_to_generate / len(neg_def_gen_template_list_noIniNeg))
        num_followup_neg_examples_adjusted = math.ceil(num_followup_neg_examples / len(gen_template_list) / self.args.num_follow_neg_rels_to_generate)
        
        # first generate the followup negative rel def prompts
        task_id = 0
        if os.path.exists(os.path.join(llm_ckpt_folder, 'neg_def_prompts_followup.pt')):
            followup_neg_def_prompts = torch.load(os.path.join(llm_ckpt_folder, 'neg_def_prompts_followup.pt'))
        else:
            if os.path.exists(os.path.join(prev_llm_ckpt_folder_neg, 'neg_def_prompts.pt')):
                followup_neg_def_prompts, task_id = self.LLM_followup_neg_def_generation_parallel(rel_set=rel_set, rel_def_prompt_list=rel_def_prompt_list, representative_relation_patterns_examples=representative_relation_patterns_examples, neg_def_gen_template_list=neg_def_gen_template_list, num_neg_rels_to_generate_adjusted=num_followup_neg_rels_to_generate_adjusted, LLM_model=LLM_model_def_gen, max_num_attempts=max_num_attempts, task_id=task_id, llm_ckpt_folder=llm_ckpt_folder, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window, prev_llm_ckpt_folder_neg=prev_llm_ckpt_folder_neg, feedback_w_scores=feedback_w_scores, representative_relation_patterns_examples_scores=representative_relation_patterns_examples_scores)
            else:
                followup_neg_def_prompts, task_id = self.LLM_followup_neg_def_generation_parallel(rel_set=rel_set, rel_def_prompt_list=rel_def_prompt_list, representative_relation_patterns_examples=representative_relation_patterns_examples, neg_def_gen_template_list=neg_def_gen_template_list_noIniNeg, num_neg_rels_to_generate_adjusted=num_followup_neg_rels_to_generate_adjusted, LLM_model=LLM_model_def_gen, max_num_attempts=max_num_attempts, task_id=task_id, llm_ckpt_folder=llm_ckpt_folder, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window, prev_llm_ckpt_folder_neg=prev_llm_ckpt_folder_neg, feedback_w_scores=feedback_w_scores, representative_relation_patterns_examples_scores=representative_relation_patterns_examples_scores)    
        
        input_list = []
        input_list_x_elt = []
        
        
        # then generate neg rel examples
        assert len(followup_neg_def_prompts) == len(rel_set)
        for rel_id, rel in enumerate(rel_set):
            rel_neg_def_prompts = followup_neg_def_prompts[rel_id]
            assert len(rel_neg_def_prompts) == self.args.num_follow_neg_rels_to_generate
            for rel_neg_def_prompt_id, rel_neg_def_prompt in enumerate(rel_neg_def_prompts):
                for gen_template_id, gen_template in enumerate(gen_template_list):
                    prompt = gen_template.substitute(rel_def_prompt=rel_neg_def_prompt, num_examples=num_followup_neg_examples_adjusted)
                    input_list.append({
                        'task_id': task_id,
                        'messages': [
                            {"role": "user", "content": prompt}
                        ]
                    })
                    
                    input_list_x_elt.append(f"followup_neg_exp_{rel_id}-{rel_neg_def_prompt_id}-{gen_template_id}")
                    task_id += 1
                    
        output_list = LLM_model.async_calls(input_list=input_list)
        current_best_output_list = deepcopy(output_list)
    
        rels_neg_examples_followup = []
        
        input_list_to_regen = []
        input_list_to_regen_x_elt = []
        
        current_num_attempts = 0
        for input_idx, (input_x_elt, output_response) in enumerate(zip(input_list_x_elt, output_list)):
            rel_id = int(input_x_elt[17:].split('-')[0])
            rel_neg_def_prompt_id = int(input_x_elt[17:].split('-')[1])
            gen_template_id = int(input_x_elt[17:].split('-')[2])
            rel = rel_set[rel_id]
            
            temp_rel_examples = chat_response_to_examples_parser(response=deepcopy(output_response), relation=f"!{rel}", num_examples=num_followup_neg_examples_adjusted)
            if len(temp_rel_examples) == num_followup_neg_examples_adjusted:
                rels_neg_examples_followup.append(deepcopy(temp_rel_examples))
            else:
                input_list_to_regen.append(deepcopy(input_list[input_idx]))
                input_list_to_regen_x_elt.append(deepcopy(input_x_elt))
                rels_neg_examples_followup.append(None)
               
                
        while (None in rels_neg_examples_followup) and current_num_attempts <= max_num_attempts:
            print(f"Current attempt: {current_num_attempts} with #invalid inputs to regenerate={rels_neg_examples_followup.count(None)}")
            assert rels_neg_examples_followup.count(None) == len(input_list_to_regen)
            
            output_list_to_regen = LLM_model.async_calls(input_list=input_list_to_regen)
            
            input_list_to_regen_temp = []
            input_list_to_regen_x_elt_temp = []
            
            for input_to_regen_idx, (input_x_elt_to_regen, output_response_to_regen) in enumerate(zip(input_list_to_regen_x_elt, output_list_to_regen)):
                rel_id = int(input_x_elt_to_regen[17:].split('-')[0])
                rel_neg_def_prompt_id = int(input_x_elt_to_regen[17:].split('-')[1])
                gen_template_id = int(input_x_elt_to_regen[17:].split('-')[2])
                rel = rel_set[rel_id]
                
                absolute_idx = input_list_x_elt.index(input_x_elt_to_regen)

                temp_rel_examples = chat_response_to_examples_parser(response=deepcopy(output_response_to_regen), relation=f"!{rel}", num_examples=deepcopy(num_followup_neg_examples_adjusted))
                
                if len(temp_rel_examples) == num_followup_neg_examples_adjusted:
                    assert rels_neg_examples_followup[absolute_idx] is None
                    rels_neg_examples_followup[absolute_idx] = deepcopy(temp_rel_examples)
                    current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
                else:
                    input_list_to_regen_temp.append(deepcopy(input_list_to_regen[input_to_regen_idx]))
                    input_list_to_regen_x_elt_temp.append(deepcopy(input_x_elt_to_regen))
                    
                    if abs(len(temp_rel_examples) - num_followup_neg_examples_adjusted) <= 2 and len(temp_rel_examples) > len(chat_response_to_examples_parser(response=deepcopy(current_best_output_list[absolute_idx]), relation=f"!{rel}", num_examples=deepcopy(num_followup_neg_examples_adjusted))):
                        current_best_output_list[absolute_idx] = deepcopy(output_response_to_regen)
            
            input_list_to_regen = deepcopy(input_list_to_regen_temp)
            input_list_to_regen_x_elt = deepcopy(input_list_to_regen_x_elt_temp)
            
            current_num_attempts += 1
            
            
        rels_neg_examples_followup_output = []
        for rel_example_idx, (input_x_elt, rel_example_list) in enumerate(zip(input_list_x_elt, rels_neg_examples_followup)):
            rel_id = int(input_x_elt[17:].split('-')[0])
            rel_neg_def_prompt_id = int(input_x_elt[17:].split('-')[1])
            gen_template_id = int(input_x_elt[17:].split('-')[2])
            rel = rel_set[rel_id]
            
            if rel_neg_def_prompt_id == 0: rels_neg_examples_followup_output.append([])
            
            if rel_example_list is not None:
                rels_neg_examples_followup_output[rel_id].extend(rel_example_list)
            else:
                rels_neg_examples_followup_output[rel_id].extend(chat_response_to_examples_parser(response=deepcopy(current_best_output_list[rel_example_idx]), relation=f"!{rel}", num_examples=deepcopy(num_followup_neg_examples_adjusted)))
                
            
            response = current_best_output_list[rel_example_idx]
            response_dump = response.model_dump()
            response_dump.update({'prompt_input': input_list[rel_example_idx], 'relation': f"!{rel}", 'relative_id': input_x_elt})
            append_to_jsonl(
                data=response_dump,
                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat_neg_exp.jsonl'),
            )
        
        
        torch.save(followup_neg_def_prompts, os.path.join(llm_ckpt_folder, 'neg_def_prompts_followup.pt'))
        if os.path.exists(os.path.join(prev_llm_ckpt_folder_neg, 'neg_def_prompts.pt')):
            prev_neg_def_prompts = torch.load(os.path.join(prev_llm_ckpt_folder_neg, 'neg_def_prompts.pt'))
            accumulated_neg_def_prompts = []
            for rel_id in range(len(prev_neg_def_prompts)):
                accumulated_neg_def_prompts.append(deepcopy(prev_neg_def_prompts[rel_id]) + deepcopy(followup_neg_def_prompts[rel_id]))
        else: accumulated_neg_def_prompts = deepcopy(followup_neg_def_prompts)
        torch.save(accumulated_neg_def_prompts, os.path.join(llm_ckpt_folder, 'neg_def_prompts.pt'))
        # above, neg_def_prompts.pt file alaways stores the so far accumulated negative relation definitions as a list of strings
        
        return rels_neg_examples_followup_output
    
    
    def LLM_followup_neg_generation(self, rel_set, rel_def_prompt_list, representative_relation_patterns_examples, num_followup_neg_examples=10, llm_ckpt_folder=None, sliding_window_size=10, enable_sliding_window=True):
        assert llm_ckpt_folder is not None, "llm_ckpt_folder not specified!"
        LLM_model = GPTCompletionModel(
            GPT_model=self.args.llm_model_ckpt,
            save_filepath=os.path.join(llm_ckpt_folder, f'gpt4_chat_neg.jsonl'),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        LLM_model.update_call_attributes(max_tokens=4096, seed=self.args.seed)
        
        max_num_attempts = 3

        gen_template_list = [
            Template("A binary relation between entity placeholders <ENT0> and <ENT1> is defined by \"${pos_rel_def_prompt}\" (in relation examples, <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0> , <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1> .). Typical examples predicted as positive by current relation extraction model are:\n\n${feedback_examples}\n\nGenerate ${num_examples} negative relation examples (numbered from 1 to ${num_examples}) following the required entity mention prefix and suffix formats. Other requirements are: 1. Ensure that generated negative examples are of different but challenging negative relations and distinguish from positive relation in terms of entities or contexts. 2. Identify model's bias from the typical predicted examples which do not express the positive relation correctly and your newly generated negative examples should try to fix the bias. 3. Try as many different relation patterns or relation expressions as possible."),
        ]
        
        rels_neg_examples_followup = []
        
        num_followup_neg_examples_adjusted = math.ceil(num_followup_neg_examples / len(gen_template_list))
        
        task_id = 0
        for rel_id, rel, rel_def_prompt in zip(range(len(rel_set)), rel_set, rel_def_prompt_list):
            rel_example_list = []
            
            rel_representative_relation_patterns_examples = representative_relation_patterns_examples[rel_id]
            feedback_examples_to_print = []
            for ex_id, ex in enumerate(rel_representative_relation_patterns_examples):
                feedback_examples_to_print.append(f"{ex_id + 1}. " + get_so_tagged_example(example=ex, compact=True))
            feedback_examples_to_print_string = "\n\n".join(feedback_examples_to_print)
            
            
            for gen_template in gen_template_list:
                prompt = gen_template.substitute(pos_rel_def_prompt=rel_def_prompt, feedback_examples=feedback_examples_to_print_string, num_examples=num_followup_neg_examples_adjusted)
                
                qualified_flag = False
                current_num_attempts = 0
                least_rel_examples = []
                least_response_dump = {}
                while qualified_flag is not True and current_num_attempts <= max_num_attempts:
                    try:
                        print(f"Attempt {current_num_attempts} for a gen template for follow up negative examples of relation {rel}")
                        current_num_attempts += 1
                        response = LLM_model(input={'task_id': task_id, 'messages': [{'role': 'user', 'content': deepcopy(prompt)}]})
                        
                        temp_rel_examples = parse_chat_response_to_examples(response=response, relation=f"!{rel}", num_examples=num_followup_neg_examples_adjusted)
                        
                        response_dump = response.model_dump()
                        response_dump.update({'prompt_input': {'task_id': task_id, 'messages': [{'role': 'user', 'content': prompt}]}, 'relation': f"!{rel}",})
                        
                        if len(temp_rel_examples) == num_followup_neg_examples_adjusted:    
                            append_to_jsonl(
                                data=response_dump,
                                filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                            )
                            rel_example_list.extend(temp_rel_examples)
                            qualified_flag = True
                        else:
                            if abs(len(temp_rel_examples) - num_followup_neg_examples_adjusted) <= 2 and len(temp_rel_examples) > len(least_rel_examples):
                                least_rel_examples = temp_rel_examples
                                least_response_dump = response_dump
                                
                    except openai.InternalServerError as e:
                        print("An Internal Server Error occurred when calling openai api:", e)
                
                
                if not qualified_flag:
                    rel_example_list.extend(least_rel_examples)
                    append_to_jsonl(
                        data=least_response_dump,
                        filename=os.path.join(llm_ckpt_folder, f'gpt4_chat.jsonl'),
                    )
                    print(f"Eventually LLM generation not qualified: Relation={rel} with #generated results={len(least_rel_examples)}. Use the sub-optimal generated results instead.")
                task_id += 1
                
            rels_neg_examples_followup.append(rel_example_list)
            print(f'Parsed negative examples for {rel}')
            self.dataloader.print_div_raw_data_w_indices(example_list=rels_neg_examples_followup[-1], indent_char='|\t')
        
        return rels_neg_examples_followup
    
    def _snowball_iterative_get_rels_neg_examples_followup(self, rels_neg_examples_followup_ckpt, llm_ckpt_folder, rel_set, rel_def_prompt_list, num_follow_neg_examples, max_paraphrased_prompts, prev_hist_evaluation_results, new_threshold, top_N, print_hierarchy=f"", cluster_patterns=False, feedback_w_scores=False, prev_llm_ckpt_folder_neg=None, sliding_window_size=10, enable_sliding_window=True,):
        if not os.path.exists(rels_neg_examples_followup_ckpt):
            prev_unlabeled_inference_results = prev_hist_evaluation_results['rel_unlabeled_inference_results']
            unlabeled_corpus_div = prev_hist_evaluation_results['unlabeled_corpus_div']

            # https://spacy.io/usage/processing-pipelines#_title for more info on spacy nlp.pipe
            spacy.prefer_gpu()
            # nlp = spacy.load("en_core_web_lg")
            nlp = spacy.load("en_core_web_trf")
            n_cores = multiprocessing.cpu_count()
            
            representative_relation_patterns_examples = []
            representative_relation_patterns_examples_scores = []
            representative_relation_patterns_examples_ckpt = os.path.join(llm_ckpt_folder, 'feedback_examples_neg.pt')
            if os.path.exists(representative_relation_patterns_examples_ckpt): representative_relation_patterns_examples = torch.load(representative_relation_patterns_examples_ckpt)
            
            
            if feedback_w_scores:
                representative_relation_patterns_examples_scores_ckpt = os.path.join(llm_ckpt_folder, 'feedback_examples_pos_scores_neg.pt')
                if os.path.exists(representative_relation_patterns_examples_scores_ckpt): representative_relation_patterns_examples_scores = torch.load(representative_relation_patterns_examples_scores_ckpt)
                
                
            feedback_intermediate_results = {
                'cluster_patterns': cluster_patterns,
                'feedback_w_scores': feedback_w_scores,
                'relation_patterns': [],
                'relation_patterns_examples': [],
                'relation_patterns_w_masks': [],
                'relation_patterns_w_masks_tagged': [],
                'cluster_assignment': [],
                'clustered_data_indices_sorted': [],
            }
            
            
            for rel_id, rel in enumerate(rel_set):
                if os.path.exists(representative_relation_patterns_examples_ckpt):
                    print(f"{print_hierarchy}\tCached feedback examples file exists. Skipping mining process for {rel} ({self.dataloader.rel_info[rel]['relation_name']})...")
                    print(f"{print_hierarchy}\tFeedback examples: ")
                    self.dataloader.print_div_raw_data_w_indices(example_list=representative_relation_patterns_examples[rel_id], indent_char='|\t')
                    continue
                
                print(f"{print_hierarchy}===Mining relation pattern feedbacks to LLM: {rel} ({self.dataloader.rel_info[rel]['relation_name']})===")
                if cluster_patterns: print(f"{print_hierarchy}|\tPattern Mining Method: pattern extraction and clustering")
                else: print(f"{print_hierarchy}|\tPattern Mining Method: random sampling")
                
                
                rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
                rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
                print(f"{print_hierarchy}|\tRelation definition prompt for {rel}: {rel_def_prompt}")
                for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                    print(f"{print_hierarchy}|\t\tParaphrased prompt {i} for {rel}: {rel_def_paraphrased_prompt}")
                
                prev_rel_hist_evaluation_results = prev_hist_evaluation_results['rel_hist_evaluation_results'][rel_id]
                prev_rel_unlabeled_inference_results = prev_unlabeled_inference_results[rel_id]
                assert (rel == prev_rel_hist_evaluation_results['rel']) and (prev_rel_unlabeled_inference_results['inference_relative_ids'].tolist() == list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data']))))

                rel_new_threshold = deepcopy(new_threshold)
                rel_top_N = deepcopy(top_N)
                print(f"{print_hierarchy}\tSetting the rel_new_threshold={rel_new_threshold} and setting rel_top_N={rel_top_N} for selecting positive instances from unlabled corpus")
                prev_rel_unlabeled_pred_probs = prev_rel_unlabeled_inference_results['pred_probs']
                prev_rel_unlabeled_inference_relative_ids = prev_rel_unlabeled_inference_results['inference_relative_ids']
                count_above_threshold = np.sum(prev_rel_unlabeled_pred_probs[:, 1] >= rel_new_threshold)
                if count_above_threshold < rel_top_N:
                    print(f"{print_hierarchy}\tNumber of pred probs >= rel_new_threshold ({count_above_threshold}) is smaller than rel_top_N. Adjusting rel_top_N to this number.")
                    rel_top_N = count_above_threshold
                sorted_indices = np.argsort(prev_rel_unlabeled_pred_probs[:, 1])[::-1] # numpy array of (|unlabeled corpus|, ), indicating the indices from largest pos pred prob to least
                prev_rel_unlabeled_inference_relative_ids_sorted = prev_rel_unlabeled_inference_relative_ids[sorted_indices] # numpy array of (|unlabeled corpus|, ), indicating the indices to the unlabeled corpus samples sorted by descending pos pred prob
                prev_top_N_unlabeled_indices = prev_rel_unlabeled_inference_relative_ids_sorted[:rel_top_N]
                prev_rel_top_unlabeled_examples = self.dataloader.get_div_raw_data_w_indices(div=unlabeled_corpus_div, div_indices=prev_top_N_unlabeled_indices)
                
                if feedback_w_scores:
                    prev_rel_top_unlabeled_examples, prev_rel_top_unlabeled_examples_pos_scores = remove_duplicated_invalid_examples_v1(examples=prev_rel_top_unlabeled_examples, scores=prev_rel_unlabeled_pred_probs[:, 1][sorted_indices[:rel_top_N]])
                else:
                    prev_rel_top_unlabeled_examples = remove_duplicated_invalid_examples(examples=prev_rel_top_unlabeled_examples)

                if cluster_patterns:
                    preprocessed_docs = list(nlp.pipe([" ".join(unlabeled_ex['tokens']) for unlabeled_ex in prev_rel_top_unlabeled_examples], batch_size=2048))

                    h_t_unconnected_count = 0 # counting the number of examples where head ent and tail ent are not in the same dep parsing tree
                    relation_patterns = []
                    relation_patterns_examples = []
                    relation_patterns_w_masks = []
                    relation_patterns_w_masks_tagged = []
                    rel_representative_relation_patterns_examples = []
                    rel_representative_relation_patterns_examples_scores = []
                    relation_patterns_original_relative_indices = []
                    
                    
                    for unlabeled_ex_id, unlabeled_ex in enumerate(tqdm(prev_rel_top_unlabeled_examples, desc='Examples')):
                        doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path = get_shortest_dep_path_v3(word_list=unlabeled_ex['tokens'], ht_word_spans=[[unlabeled_ex['h'][2][0][0], unlabeled_ex['h'][2][0][-1]], [unlabeled_ex['t'][2][0][0], unlabeled_ex['t'][2][0][-1]]], doc_given=preprocessed_docs[unlabeled_ex_id])

                        if h_doc_span_path is None:
                            h_t_unconnected_count += 1
                            continue
                        
                        h_dep_t_string = " ".join(linearize_ent_dep_path(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, add_tags=False, add_h_t=True))
                        h_dep_t_string_w_mask = " ".join(linearize_ent_dep_path_w_mask(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, mask_h_t=False, mask_token='<mask>', tag_h_t=False))
                        h_dep_t_string_w_mask_tagged = " ".join(linearize_ent_dep_path_w_mask(doc=doc, h_doc_span=h_doc_span, t_doc_span=t_doc_span, dependency_path=dependency_path, mask_h_t=False, mask_token='<mask>', tag_h_t=True))
                        
                        relation_patterns_original_relative_indices.append(unlabeled_ex_id)
                        relation_patterns.append(h_dep_t_string)
                        relation_patterns_examples.append(unlabeled_ex)
                        relation_patterns_w_masks.append(h_dep_t_string_w_mask)
                        relation_patterns_w_masks_tagged.append(h_dep_t_string_w_mask_tagged)
                        
                
                    relation_patterns_embs = get_SBERT_embedding(sent_list=relation_patterns_w_masks, sbert_model='all-mpnet-base-v2', batch_size=768, normalize_embeddings=True)

                    cluster_centroids, cluster_assignment, clustered_data_indices, clustered_dists, dist = get_KMeans_clusters(data_embeddings=relation_patterns_embs, seed=self.args.seed, n_clusters=10)
                    clustered_data_indices_sorted = []
                    for centroid_idx in range(len(clustered_data_indices)):
                        assert len(clustered_data_indices[centroid_idx]) == len(clustered_dists[centroid_idx])
                        clustered_data_indices_sorted.append([])
                        sorted_indices = np.argsort(clustered_dists[centroid_idx]) # should be reversed for using HDBSCAN
                        for j in sorted_indices: clustered_data_indices_sorted[-1].append(clustered_data_indices[centroid_idx][j])
                    for centroid_idx in range(len(clustered_data_indices_sorted)):
                        rel_representative_relation_patterns_examples.append(relation_patterns_examples[clustered_data_indices_sorted[centroid_idx][0]])

                        if feedback_w_scores:
                            rel_representative_relation_patterns_examples_scores.append(float(prev_rel_top_unlabeled_examples_pos_scores[relation_patterns_original_relative_indices[clustered_data_indices_sorted[centroid_idx][0]]]))
                            
                    for centroid_idx, clustered_data_ids in enumerate(clustered_data_indices_sorted):
                        print(f'{print_hierarchy}\nCluster {centroid_idx} and its top examples')
                        for j in clustered_data_ids[:20]:
                            print("\t\tPattern: " + relation_patterns_w_masks_tagged[j])
                            print("\t\tExample:" + get_so_tagged_example(example=relation_patterns_examples[j], compact=True))
                            print('--' * 30)
                        print("\n")


                    feedback_intermediate_results['relation_patterns'].append(relation_patterns)
                    feedback_intermediate_results['relation_patterns_examples'].append(relation_patterns_examples)
                    feedback_intermediate_results['relation_patterns_w_masks'].append(relation_patterns_w_masks)
                    feedback_intermediate_results['relation_patterns_w_masks_tagged'].append(relation_patterns_w_masks_tagged)
                    feedback_intermediate_results['cluster_assignment'].append(cluster_assignment)
                    feedback_intermediate_results['clustered_data_indices_sorted'].append(clustered_data_indices_sorted)

                else:
                    sample_num = 30
                    if feedback_w_scores:
                        rel_representative_relation_patterns_examples_ids = random.sample(range(prev_rel_top_unlabeled_examples), k=sample_num)
                        rel_representative_relation_patterns_examples = [prev_rel_top_unlabeled_examples[rel_representative_relation_patterns_examples_i] for rel_representative_relation_patterns_examples_i in rel_representative_relation_patterns_examples_ids]
                        rel_representative_relation_patterns_examples_scores = [float(prev_rel_top_unlabeled_examples_pos_scores[rel_representative_relation_patterns_examples_i]) for rel_representative_relation_patterns_examples_i in rel_representative_relation_patterns_examples_ids]
                    else:
                        rel_representative_relation_patterns_examples = random.sample(prev_rel_top_unlabeled_examples, k=sample_num)
                    
                    
                representative_relation_patterns_examples.append(rel_representative_relation_patterns_examples)

                if feedback_w_scores:
                    representative_relation_patterns_examples_scores.append(rel_representative_relation_patterns_examples_scores)
                    
            if not os.path.exists(representative_relation_patterns_examples_ckpt):
                torch.save(representative_relation_patterns_examples, representative_relation_patterns_examples_ckpt)
                torch.save(feedback_intermediate_results, os.path.join(llm_ckpt_folder, 'feedback_intermediate_results_neg.pt'))

            if feedback_w_scores and not os.path.exists(representative_relation_patterns_examples_scores_ckpt):
                torch.save(representative_relation_patterns_examples_scores, representative_relation_patterns_examples_scores_ckpt)
                
            # representative_relation_patterns_examples = torch.load(representative_relation_patterns_examples_ckpt)
            
            rels_neg_examples_followup = self.LLM_followup_neg_generation_parallel(rel_set=rel_set, rel_def_prompt_list=rel_def_prompt_list, representative_relation_patterns_examples=representative_relation_patterns_examples, num_followup_neg_examples=num_follow_neg_examples, llm_ckpt_folder=llm_ckpt_folder, prev_llm_ckpt_folder_neg=prev_llm_ckpt_folder_neg, feedback_w_scores=feedback_w_scores, representative_relation_patterns_examples_scores=representative_relation_patterns_examples_scores, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window)
            torch.save(rels_neg_examples_followup, rels_neg_examples_followup_ckpt)

        return None

    def _run_snowball_iterative_feedbacktrain(self, snowball_iter_id, pos_neg_suffix_to_read, pos_neg_suffix_to_save, num_follow_pos_examples, rel_set, eval_locIds, max_paraphrased_prompts, ckpt_sub_folder, llm_ckpt_sub_folder, prev_llm_ckpt_sub_folder, prev_hist_evaluation_results, new_threshold=0.85, top_N=50000, neg_sample_strategy='random_all', neg_pos_ratio=2, deterministic=True, prev_chosen_model_state_dict_strategy=None, run_snowball=True, prev_ckpt_sub_folder=None, random_seed_multiplier=1):
        """
        Args:

        
        """
        print_hierarchy = f"[Iter-{snowball_iter_id}/FeedbackTrain]: Set threshold={new_threshold}, top_N={top_N}, neg_pos_ratio={neg_pos_ratio}"


        prev_llm_ckpt_folder = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, prev_llm_ckpt_sub_folder)
        llm_ckpt_folder = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, llm_ckpt_sub_folder)
        os.makedirs(llm_ckpt_folder, exist_ok=True)
        os.makedirs(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder), exist_ok=True)
        rels_pos_examples_followup_ckpt = os.path.join(llm_ckpt_folder, f'rel_init_pos.pt')


        prev_unlabeled_inference_results = prev_hist_evaluation_results['rel_unlabeled_inference_results']
        test_div = prev_hist_evaluation_results['test_div']
        unlabeled_corpus_div = prev_hist_evaluation_results['unlabeled_corpus_div']


        hist_evaluation_results = {
            'test_div': test_div,
            'unlabeled_corpus_div': unlabeled_corpus_div,
            'args': self.args,
            'rel_set': rel_set,
            'eval_locIds': eval_locIds,
            'rel_hist_evaluation_results': [],
            'rel_unlabeled_inference_results': [],
            'train_inference_ckpt_exists': []
        }

        # gathering feedback for continual LLM pos generation
        self._snowball_iterative_get_rels_pos_examples_followup(rels_pos_examples_followup_ckpt=rels_pos_examples_followup_ckpt, prev_llm_ckpt_folder=prev_llm_ckpt_folder, llm_ckpt_folder=llm_ckpt_folder, rel_set=rel_set, max_paraphrased_prompts=max_paraphrased_prompts, prev_hist_evaluation_results=prev_hist_evaluation_results, num_follow_pos_examples=num_follow_pos_examples, new_threshold=new_threshold, top_N=top_N, print_hierarchy=f"[Iter-{snowball_iter_id}/FeedbackTrain/FollowupPosGen]: ")
        rels_pos_examples_followup = torch.load(rels_pos_examples_followup_ckpt)
        writer = SummaryWriter(log_dir=os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, 'tensorboard/'))


        # use feedback to train RE model
        for rel_id, rel in enumerate(rel_set):
            print(f"{print_hierarchy}===Training with further generated examples on relation: {rel} ({self.dataloader.rel_info[rel]['relation_name']})===")

            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
            print(f"{print_hierarchy}|\tRelation definition prompt for {rel}: {rel_def_prompt}")
            for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                print(f"{print_hierarchy}|\t\tParaphrased prompt {i} for {rel}: {rel_def_paraphrased_prompt}")

            
            prev_rel_hist_evaluation_results = prev_hist_evaluation_results['rel_hist_evaluation_results'][rel_id]
            prev_rel_unlabeled_inference_results = prev_unlabeled_inference_results[rel_id]
            assert rel == prev_rel_hist_evaluation_results['rel'] and prev_rel_unlabeled_inference_results['inference_relative_ids'].tolist() == list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data'])))


            rel_pos_examples = rels_pos_examples_followup[rel_id]
            print(f"{print_hierarchy}LLM additionally generated positive examples after receiving feedbacks: ")
            self.dataloader.print_div_raw_data_w_indices(div_list=None, local_indices=None, indent_char='\t', example_list=rel_pos_examples)


            self._snowball_iterative_model_train_inference(rel_id=rel_id, rel=rel, rel_pos_examples=rel_pos_examples, ckpt_sub_folder=ckpt_sub_folder, pos_neg_suffix_to_read=pos_neg_suffix_to_read, pos_neg_suffix_to_save=pos_neg_suffix_to_save, hist_evaluation_results=hist_evaluation_results, prev_rel_hist_evaluation_results=prev_rel_hist_evaluation_results, prev_rel_unlabeled_inference_results=prev_rel_unlabeled_inference_results, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, prev_chosen_model_state_dict_strategy=prev_chosen_model_state_dict_strategy, test_div=test_div, eval_locIds=eval_locIds, run_snowball=run_snowball, deterministic=deterministic, neg_sample_strategy=neg_sample_strategy, unlabeled_corpus_div=unlabeled_corpus_div, neg_pos_ratio=neg_pos_ratio, writer=writer, print_hierarchy=print_hierarchy, prev_ckpt_sub_folder=prev_ckpt_sub_folder, random_seed_multiplier=random_seed_multiplier)


            
        for epoch_idx in self.args.save_epochs:
            print(f"{print_hierarchy}Results (at epoch={epoch_idx}) averaged over all relations: P: {np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")

        print(f"{print_hierarchy}Chosen results averaged over all relations: P: {np.mean([i['chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")


        for epoch_idx in range(self.args.num_train_epochs):
            avg_precision = np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged f1", avg_f1, epoch_idx)

        gc.collect()
        torch.cuda.empty_cache()
        
        for rel_id, rel in enumerate(rel_set):
            rel_ckpt_exists = hist_evaluation_results['train_inference_ckpt_exists'][rel_id]
            rel_hist_evaluation_results = hist_evaluation_results['rel_hist_evaluation_results'][rel_id]

            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]

            self._snowball_iterative_model_unlabel_inference(rel=rel, ckpt_sub_folder=ckpt_sub_folder, hist_evaluation_results=hist_evaluation_results, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, run_snowball=run_snowball, rel_hist_evaluation_results=rel_hist_evaluation_results, unlabeled_corpus_div=unlabeled_corpus_div, print_hierarchy=print_hierarchy, ckpt_exists=rel_ckpt_exists)
            
            
        self.save_model(
            os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_all.pt'),
            hist_evaluation_results,
        )

        return hist_evaluation_results
    

    def run_snowball_iterative_main(self, test_div='test', unlabeled_corpus_div='distant', save_suffix=None, deterministic=True, iterative_version='v0'):
        """
        Need to correct: 1. check overall codes. 2. improve the feedback prompts

        
        Assume: all sub_folder string have '/' at the end
        """
        # some hyperparams for llm generation
        num_init_pos_examples = 30 # total init pos samples generated per relation (regardless of #generation prompts per relation)
        num_init_neg_examples = 60 # total init neg samples sampled per relation (regardless of #generation prompts per relation)
        num_follow_pos_examples = 30 # total new pos samples to generate each iteration round per relation (regardless of #generation prompts per relation)
        num_follow_neg_examples = 60 # total new neg samples to sample each iteration round per relation (regardless of #generation prompts per relation)
        

        # some other hyperparams related to snowball
        num_snowball_iterations = 1
        max_paraphrased_prompts = 0 # use at most this number of paraphrased rel def prompts
        selftrain_new_threshold, selftrain_top_N = 0.90, 500
        feedbacktrain_new_threshold, feedbacktrain_top_N = 0.95, 500
        neg_sample_strategy, neg_pos_ratio = 'random_all', 2
        # neg_sample_strategy, neg_pos_ratio = 'random_prior', 2

        print(f'Negative sampling strategy: {neg_sample_strategy}')

        if save_suffix is None: save_suffix = f'_{num_init_pos_examples}p{num_init_neg_examples}n'
        ckpt_sub_folder_step0 = f'simi_optim_NLIBased_snowball_ckpt{save_suffix}/'
        ckpt_sub_folder_iterative = f'Snowball_ckpt_{iterative_version}{save_suffix}_{num_follow_pos_examples}p{num_follow_neg_examples}nfollow/'
        llm_ckpt_sub_folder_step0 = f'llm_{num_init_pos_examples}p/'
        llm_ckpt_sub_folder_iterative = f'llm_{iterative_version}_{num_init_pos_examples}p_{num_follow_pos_examples}pfollow/'



        prev_hist_evaluation_results = torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder_step0, 'result_all.pt'))
        assert prev_hist_evaluation_results['rel'] == prev_hist_evaluation_results['rel_set']
        rel_set = prev_hist_evaluation_results['rel_set']
        eval_locIds = prev_hist_evaluation_results['eval_locIds']

        pos_neg_suffix_to_read = ['_0']
        prev_chosen_model_state_dict_strategy = 'last_save_epoch'
        prev_ckpt_sub_folder = ckpt_sub_folder_step0
        ckpt_sub_folder = None
        prev_llm_ckpt_sub_folder = llm_ckpt_sub_folder_step0
        llm_ckpt_sub_folder = None
        random_seed_multiplier = 10

        for snowball_iter_id in range(num_snowball_iterations):
            print(f"===Running Iterative Snowball. Iteration {snowball_iter_id}===")


            ### run feedback LLM for training with new instances
            pos_neg_suffix_to_save = f'_iter-{snowball_iter_id}_step-1_feedbacktrain'
            llm_ckpt_sub_folder = f'{llm_ckpt_sub_folder_iterative}iter-{snowball_iter_id}_step-1_feedbacktrain/'
            ckpt_sub_folder = f'{ckpt_sub_folder_iterative}iter-{snowball_iter_id}_step-1_feedbacktrain/'
            
            prev_hist_evaluation_results = self._run_snowball_iterative_feedbacktrain(snowball_iter_id=snowball_iter_id, pos_neg_suffix_to_read=pos_neg_suffix_to_read, pos_neg_suffix_to_save=pos_neg_suffix_to_save, num_follow_pos_examples=num_follow_pos_examples, rel_set=rel_set, eval_locIds=eval_locIds, max_paraphrased_prompts=max_paraphrased_prompts, ckpt_sub_folder=ckpt_sub_folder, llm_ckpt_sub_folder=llm_ckpt_sub_folder, prev_llm_ckpt_sub_folder=prev_llm_ckpt_sub_folder, prev_hist_evaluation_results=prev_hist_evaluation_results, new_threshold=feedbacktrain_new_threshold, top_N=feedbacktrain_top_N, neg_sample_strategy=neg_sample_strategy, neg_pos_ratio=neg_pos_ratio, deterministic=deterministic, prev_chosen_model_state_dict_strategy=None, run_snowball=True, prev_ckpt_sub_folder=prev_ckpt_sub_folder, random_seed_multiplier=random_seed_multiplier)

            pos_neg_suffix_to_read.append(pos_neg_suffix_to_save)
            prev_llm_ckpt_sub_folder = llm_ckpt_sub_folder
            prev_ckpt_sub_folder = ckpt_sub_folder
            random_seed_multiplier += 1



            ### run "self training" to consolidate self knowledge
            pos_neg_suffix_to_save = f'_iter-{snowball_iter_id}_step-2_selftrain'
            ckpt_sub_folder = f'{ckpt_sub_folder_iterative}iter-{snowball_iter_id}_step-2_selftrain/'

            prev_hist_evaluation_results = self._run_snowball_iterative_selftrain(snowball_iter_id=snowball_iter_id, pos_neg_suffix_to_read=pos_neg_suffix_to_read, pos_neg_suffix_to_save=pos_neg_suffix_to_save, rel_set=rel_set, eval_locIds=eval_locIds, max_paraphrased_prompts=max_paraphrased_prompts, ckpt_sub_folder=ckpt_sub_folder, prev_hist_evaluation_results=prev_hist_evaluation_results, new_threshold=selftrain_new_threshold, top_N=selftrain_top_N, neg_sample_strategy=neg_sample_strategy, neg_pos_ratio=neg_pos_ratio, deterministic=deterministic, prev_chosen_model_state_dict_strategy=prev_chosen_model_state_dict_strategy, run_snowball=False, prev_ckpt_sub_folder=prev_ckpt_sub_folder, random_seed_multiplier=random_seed_multiplier)

            # pos_neg_suffix_to_read.append(pos_neg_suffix_to_save)
            prev_ckpt_sub_folder = ckpt_sub_folder
            random_seed_multiplier += 1


    def _run_snowball_iterative_feedbacktrain_v1(self, snowball_iter_id, pos_neg_suffix_to_read, pos_neg_suffix_to_save, num_follow_pos_examples, rel_set, eval_locIds, max_paraphrased_prompts, ckpt_sub_folder, llm_ckpt_sub_folder, prev_hist_evaluation_results, new_threshold_pos, top_N_pos,  new_threshold_neg, top_N_neg, neg_sample_strategy='random_all', neg_pos_ratio=2, deterministic=True, prev_chosen_model_state_dict_strategy=None, run_snowball=True, prev_ckpt_sub_folder=None, random_seed_multiplier=1, prev_llm_ckpt_sub_folder_pos=None, prev_llm_ckpt_sub_folder_neg=None, sliding_window_size=10, enable_sliding_window=True, cluster_patterns=False, feedback_w_scores=False):
        """
        Args:

        
        """
        print_hierarchy = f"[Iter-{snowball_iter_id}/FeedbackTrain]: Set threshold pos={new_threshold_pos}, top_N pos={top_N_pos}, threshold neg={new_threshold_neg}, top_N neg={top_N_neg}"


        prev_llm_ckpt_folder_pos = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, prev_llm_ckpt_sub_folder_pos)
        prev_llm_ckpt_folder_neg = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, prev_llm_ckpt_sub_folder_neg)
        
        llm_ckpt_folder = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, llm_ckpt_sub_folder)
        os.makedirs(llm_ckpt_folder, exist_ok=True)
        os.makedirs(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder), exist_ok=True)
        rels_pos_examples_followup_ckpt = os.path.join(llm_ckpt_folder, f'rel_init_pos.pt')
        rels_neg_examples_followup_ckpt = os.path.join(llm_ckpt_folder, f'rel_init_neg.pt')

        
        rels_dev_pos_examples_followup_ckpt = os.path.join(llm_ckpt_folder, f'rel_dev_pos_0.pt')
        
        prev_hist_evaluation_results['rel_unlabeled_inference_results'] = [remove_repeats_based_on_inference_ids(data_dict=elt) for elt in prev_hist_evaluation_results['rel_unlabeled_inference_results']]
        prev_unlabeled_inference_results = prev_hist_evaluation_results['rel_unlabeled_inference_results']
        test_div = prev_hist_evaluation_results['test_div']
        unlabeled_corpus_div = prev_hist_evaluation_results['unlabeled_corpus_div']


        hist_evaluation_results = {
            'test_div': test_div,
            'unlabeled_corpus_div': unlabeled_corpus_div,
            'args': self.args,
            'rel_set': rel_set,
            'eval_locIds': eval_locIds,
            'rel_hist_evaluation_results': [],
            'rel_unlabeled_inference_results': [],
            'train_inference_ckpt_exists': []
        }


        # gathering feedback for continual LLM pos generation
        self._snowball_iterative_get_rels_pos_examples_followup(rels_pos_examples_followup_ckpt=rels_pos_examples_followup_ckpt, prev_llm_ckpt_folder=prev_llm_ckpt_folder_pos, llm_ckpt_folder=llm_ckpt_folder, rel_set=rel_set, max_paraphrased_prompts=max_paraphrased_prompts, prev_hist_evaluation_results=prev_hist_evaluation_results, num_follow_pos_examples=num_follow_pos_examples, new_threshold=new_threshold_pos, top_N=top_N_pos, print_hierarchy=f"[Iter-{snowball_iter_id}/FeedbackTrain/FollowupPosGen]: ", cluster_patterns=cluster_patterns, feedback_w_scores=feedback_w_scores, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window, rels_dev_pos_examples_followup_ckpt=rels_dev_pos_examples_followup_ckpt)
        rels_pos_examples_followup = torch.load(rels_pos_examples_followup_ckpt)
        
        rels_dev_pos_examples_followup = torch.load(rels_dev_pos_examples_followup_ckpt)
        
        if self.args.run_neg_follow_gen:
            self._snowball_iterative_get_rels_neg_examples_followup(rels_neg_examples_followup_ckpt=rels_neg_examples_followup_ckpt, llm_ckpt_folder=llm_ckpt_folder, rel_set=rel_set, rel_def_prompt_list=[self.dataloader.rel_info[r]["typed_desc_prompt"] for r in rel_set], num_follow_neg_examples=self.args.num_follow_neg_examples_to_generate,  max_paraphrased_prompts=max_paraphrased_prompts, prev_hist_evaluation_results=prev_hist_evaluation_results, new_threshold=new_threshold_neg, top_N=top_N_neg, print_hierarchy=f"[Iter-{snowball_iter_id}/FeedbackTrain/FollowupNegGen]: ", cluster_patterns=cluster_patterns, feedback_w_scores=feedback_w_scores, prev_llm_ckpt_folder_neg=prev_llm_ckpt_folder_neg, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window,)
            rels_neg_examples_followup = torch.load(rels_neg_examples_followup_ckpt)
        else:
            rels_neg_examples_followup = None
        
        
        writer = SummaryWriter(log_dir=os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, 'tensorboard/'))


        # use feedback to train RE model
        for rel_id, rel in enumerate(rel_set):
            print(f"{print_hierarchy}===Training with further generated examples on relation: {rel} ({self.dataloader.rel_info[rel]['relation_name']})===")

            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]
            print(f"{print_hierarchy}|\tRelation definition prompt for {rel}: {rel_def_prompt}")
            for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                print(f"{print_hierarchy}|\t\tParaphrased prompt {i} for {rel}: {rel_def_paraphrased_prompt}")

            
            prev_rel_hist_evaluation_results = prev_hist_evaluation_results['rel_hist_evaluation_results'][rel_id]
            prev_rel_unlabeled_inference_results = prev_unlabeled_inference_results[rel_id]
            assert (rel == prev_rel_hist_evaluation_results['rel']) and (prev_rel_unlabeled_inference_results['inference_relative_ids'].tolist() == list(range(len(self.dataloader.map_processed[unlabeled_corpus_div]['raw_data']))))


            rel_pos_examples = rels_pos_examples_followup[rel_id]
            print(f"{print_hierarchy}LLM additionally generated positive examples after receiving feedbacks: ")
            self.dataloader.print_div_raw_data_w_indices(div_list=None, local_indices=None, indent_char='\t', example_list=rel_pos_examples)

            rel_dev_pos_examples = rels_dev_pos_examples_followup[rel_id]

            if rels_neg_examples_followup is None: 
                rel_neg_examples_followup = None
            else:
                rel_neg_examples_followup = rels_neg_examples_followup[rel_id]
                
            self._snowball_iterative_model_train_inference(rel_id=rel_id, rel=rel, rel_pos_examples=rel_pos_examples, ckpt_sub_folder=ckpt_sub_folder, pos_neg_suffix_to_read=pos_neg_suffix_to_read, pos_neg_suffix_to_save=pos_neg_suffix_to_save, hist_evaluation_results=hist_evaluation_results, prev_rel_hist_evaluation_results=prev_rel_hist_evaluation_results, prev_rel_unlabeled_inference_results=prev_rel_unlabeled_inference_results, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, prev_chosen_model_state_dict_strategy=prev_chosen_model_state_dict_strategy, test_div=test_div, eval_locIds=eval_locIds, run_snowball=run_snowball, deterministic=deterministic, neg_sample_strategy=neg_sample_strategy, unlabeled_corpus_div=unlabeled_corpus_div, neg_pos_ratio=neg_pos_ratio, writer=writer, print_hierarchy=print_hierarchy, prev_ckpt_sub_folder=prev_ckpt_sub_folder, random_seed_multiplier=random_seed_multiplier, rel_neg_examples_followup=rel_neg_examples_followup, rel_dev_pos_examples=rel_dev_pos_examples)
            


            
        for epoch_idx in self.args.save_epochs:
            print(f"{print_hierarchy}Results (at epoch={epoch_idx}) averaged over all relations: P: {np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")

        print(f"{print_hierarchy}Chosen results averaged over all relations: P: {np.mean([i['chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")


        for epoch_idx in range(self.args.num_train_epochs):
            avg_precision = np.mean([i['precision'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['recall'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['f1'][i['eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged f1", avg_f1, epoch_idx)

            
            avg_precision = np.mean([i['dev_precision'][i['dev_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['dev_recall'][i['dev_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['dev_f1'][i['dev_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged dev precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev f1", avg_f1, epoch_idx)
            
            
            avg_precision = np.mean([i['dev_1_precision'][i['dev_1_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['dev_1_recall'][i['dev_1_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['dev_1_f1'][i['dev_1_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged dev_1 precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev_1 recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev_1 f1", avg_f1, epoch_idx)
            
            avg_precision = np.mean([i['dev_2_precision'][i['dev_2_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_recall = np.mean([i['dev_2_recall'][i['dev_2_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            avg_f1 = np.mean([i['dev_2_f1'][i['dev_2_eval_epochs'].index(epoch_idx)] for i in hist_evaluation_results['rel_hist_evaluation_results']])
            writer.add_scalar(f"All Relations/averaged dev_2 precision", avg_precision, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev_2 recall", avg_recall, epoch_idx)
            writer.add_scalar(f"All Relations/averaged dev_2 f1", avg_f1, epoch_idx)
            

        dev_local_chosen_precision = np.mean([i['dev_chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_recall = np.mean([i['dev_chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_f1 = np.mean([i['dev_chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        writer.add_scalar(f"All Relations/dev local chosen precision", dev_local_chosen_precision, 0)
        writer.add_scalar(f"All Relations/dev local chosen recall", dev_local_chosen_recall, 0)
        writer.add_scalar(f"All Relations/dev local chosen f1", dev_local_chosen_f1, 0)
        
        dev_local_chosen_precision = np.mean([i['dev_1_chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_recall = np.mean([i['dev_1_chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_f1 = np.mean([i['dev_1_chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        writer.add_scalar(f"All Relations/dev_1 local chosen precision", dev_local_chosen_precision, 0)
        writer.add_scalar(f"All Relations/dev_1 local chosen recall", dev_local_chosen_recall, 0)
        writer.add_scalar(f"All Relations/dev_1 local chosen f1", dev_local_chosen_f1, 0)
        
        dev_local_chosen_precision = np.mean([i['dev_2_chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_recall = np.mean([i['dev_2_chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        dev_local_chosen_f1 = np.mean([i['dev_2_chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])
        writer.add_scalar(f"All Relations/dev_2 local chosen precision", dev_local_chosen_precision, 0)
        writer.add_scalar(f"All Relations/dev_2 local chosen recall", dev_local_chosen_recall, 0)
        writer.add_scalar(f"All Relations/dev_2 local chosen f1", dev_local_chosen_f1, 0)
        
        
        gc.collect()
        torch.cuda.empty_cache()
        
        for rel_id, rel in enumerate(rel_set):
            rel_ckpt_exists = hist_evaluation_results['train_inference_ckpt_exists'][rel_id]
            rel_hist_evaluation_results = hist_evaluation_results['rel_hist_evaluation_results'][rel_id]

            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts'][:max_paraphrased_prompts]

            self._snowball_iterative_model_unlabel_inference(rel=rel, ckpt_sub_folder=ckpt_sub_folder, hist_evaluation_results=hist_evaluation_results, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, run_snowball=run_snowball, rel_hist_evaluation_results=rel_hist_evaluation_results, unlabeled_corpus_div=unlabeled_corpus_div, print_hierarchy=print_hierarchy, ckpt_exists=rel_ckpt_exists)
            
            
        self.save_model(
            os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_all.pt'),
            hist_evaluation_results,
        )

        return hist_evaluation_results
    

    def run_snowball_iterative_main_v1(self, test_div='test', unlabeled_corpus_div='distant', save_suffix=None, deterministic=True, iterative_version='v0'):
        # some hyperparams for llm generation
        num_init_pos_examples = self.args.num_init_pos_examples # total init pos samples generated per relation (regardless of #generation prompts per relation)
        num_init_neg_examples = self.args.num_init_neg_examples # total init neg samples sampled per relation (regardless of #generation prompts per relation)
        num_follow_pos_examples = self.args.num_follow_pos_examples # total new pos samples to generate each iteration round per relation (regardless of #generation prompts per relation)
        num_follow_neg_examples = self.args.num_follow_neg_examples # total new neg samples to sample each iteration round per relation (regardless of #generation prompts per relation)
        
        
        # some other hyperparams related to snowball
        num_snowball_iterations = 1
        max_paraphrased_prompts = 0 # use at most this number of paraphrased rel def prompts
        selftrain_new_threshold, selftrain_top_N = 0.90, 500
        feedbacktrain_new_threshold_pos, feedbacktrain_top_N_pos = 0.85, 50000
        feedbacktrain_new_threshold_neg, feedbacktrain_top_N_neg = 0.50, 50000
        neg_sample_strategy, neg_pos_ratio = 'random_all', 2
        sliding_window_size = 10
        enable_sliding_window = True
        cluster_patterns = False
        feedback_w_scores = False
        
        
        print(f'feedbacktrain_new_threshold_pos: {feedbacktrain_new_threshold_pos} | feedbacktrain_top_N_pos: {feedbacktrain_top_N_pos} | feedbacktrain_new_threshold_neg: {feedbacktrain_new_threshold_neg} | feedbacktrain_top_N_neg: {feedbacktrain_top_N_neg} | Negative sampling strategy: {neg_sample_strategy}')
        
        # here save_suffix is for init
        init_extra_save_suffix = ''
        if save_suffix is None: 
            if self.args.run_neg_init_gen: init_extra_save_suffix = '_iniNegGen'
            save_suffix = f'_{num_init_pos_examples}p{num_init_neg_examples}n{init_extra_save_suffix}'
            
        ckpt_sub_folder_step0 = f'snowball_ckpt{save_suffix}_seed{self.args.seed}/'
        llm_ckpt_sub_folder_pos_step0 = f'llm_{num_init_pos_examples}p_seed{self.args.seed}/'
        llm_ckpt_sub_folder_neg_step0 = f'llm_{self.args.num_init_neg_examples_to_generate}n{init_extra_save_suffix}_seed{self.args.seed}/'
        
        if self.args.run_neg_follow_gen: follow_extra_save_suffix = '_folNegGen'
        else: follow_extra_save_suffix = ''
        follow_save_suffix = f'_{num_follow_pos_examples}p{num_follow_neg_examples}nfollow{follow_extra_save_suffix}'
        ckpt_sub_folder_iterative = f'snowball_iter_ckpt_{iterative_version}{save_suffix}{follow_save_suffix}_seed{self.args.seed}/'
        llm_ckpt_sub_folder_iterative = f'llm_{iterative_version}{follow_save_suffix}_seed{self.args.seed}/'
        
        prev_hist_evaluation_results = torch.load(os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder_step0, 'result_all.pt'))
        assert prev_hist_evaluation_results['rel'] == prev_hist_evaluation_results['rel_set']
        rel_set = prev_hist_evaluation_results['rel_set']
        eval_locIds = prev_hist_evaluation_results['eval_locIds']
        
        pos_neg_suffix_to_read = ['_0']
        prev_chosen_model_state_dict_strategy = None # int / 'last_save_epoch'
        prev_ckpt_sub_folder = deepcopy(ckpt_sub_folder_step0)
        ckpt_sub_folder = None
        prev_llm_ckpt_sub_folder_pos = deepcopy(llm_ckpt_sub_folder_pos_step0)
        prev_llm_ckpt_sub_folder_neg = deepcopy(llm_ckpt_sub_folder_neg_step0)
        llm_ckpt_sub_folder = None
        random_seed_multiplier = 10
        
        for snowball_iter_id in range(num_snowball_iterations):
            print(f"===Running Iterative Snowball. Iteration {snowball_iter_id}===")
            
            
            ### run feedback LLM for training with new instances
            pos_neg_suffix_to_save = f'_iter-{snowball_iter_id}_step-1_feedbacktrain'
            llm_ckpt_sub_folder = f'{llm_ckpt_sub_folder_iterative}iter-{snowball_iter_id}_step-1_feedbacktrain/'
            ckpt_sub_folder = f'{ckpt_sub_folder_iterative}iter-{snowball_iter_id}_step-1_feedbacktrain/'
            
            prev_hist_evaluation_results = self._run_snowball_iterative_feedbacktrain_v1(snowball_iter_id=snowball_iter_id, pos_neg_suffix_to_read=pos_neg_suffix_to_read, pos_neg_suffix_to_save=pos_neg_suffix_to_save, num_follow_pos_examples=num_follow_pos_examples, rel_set=rel_set, eval_locIds=eval_locIds, max_paraphrased_prompts=max_paraphrased_prompts, ckpt_sub_folder=ckpt_sub_folder, llm_ckpt_sub_folder=llm_ckpt_sub_folder, prev_hist_evaluation_results=prev_hist_evaluation_results, new_threshold_pos=feedbacktrain_new_threshold_pos, top_N_pos=feedbacktrain_top_N_pos, new_threshold_neg=feedbacktrain_new_threshold_neg, top_N_neg=feedbacktrain_top_N_neg, neg_sample_strategy=neg_sample_strategy, neg_pos_ratio=neg_pos_ratio, deterministic=deterministic, prev_chosen_model_state_dict_strategy=prev_chosen_model_state_dict_strategy, run_snowball=self.args.run_snowball, prev_ckpt_sub_folder=prev_ckpt_sub_folder, random_seed_multiplier=random_seed_multiplier, prev_llm_ckpt_sub_folder_pos=prev_llm_ckpt_sub_folder_pos, prev_llm_ckpt_sub_folder_neg=prev_llm_ckpt_sub_folder_neg, sliding_window_size=sliding_window_size, enable_sliding_window=enable_sliding_window, cluster_patterns=cluster_patterns, feedback_w_scores=feedback_w_scores)
            
            
            pos_neg_suffix_to_read.append(deepcopy(pos_neg_suffix_to_save))
            prev_llm_ckpt_sub_folder_pos = deepcopy(llm_ckpt_sub_folder)
            prev_llm_ckpt_sub_folder_neg = deepcopy(llm_ckpt_sub_folder)
            prev_ckpt_sub_folder = deepcopy(ckpt_sub_folder)
            random_seed_multiplier += 1
            
            

            
###### run_snowball_iterative ######
    def NLIBased_similarity_measure_evaluate(self, model, eval_dataloader, threshold=0.7):
        """
        Evaluation function for the NLIBased model.

        Args: 

        Returns:

        """

        # test the model
        print(f'   Start evaluating the model...')
        print(f'   Total test batch size: {eval_dataloader.batch_size}')

        eval_loss = 0.0
        nb_eval_steps = 0
        pred_probs = []
        out_label_ids = []
        gathered_logits_by_rel_prompt = []

        def similarity_criterion(positive_probs, ground_truth_labels):
            return - (positive_probs * ground_truth_labels).sum()

        model.eval()
        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'assigned_labels': batch[2]
                }

                example2rel_similarity, logits_by_rel_prompt = model(**inputs)

                loss = similarity_criterion(positive_probs=example2rel_similarity, ground_truth_labels=batch[2])

                eval_loss += loss.item()
            
            nb_eval_steps += 1

            pred_probs.append(example2rel_similarity.detach().cpu().numpy())
            out_label_ids.append(inputs['assigned_labels'].detach().cpu().numpy())
            gathered_logits_by_rel_prompt.append(logits_by_rel_prompt.detach().cpu().numpy())

        pred_probs = np.concatenate(pred_probs, axis=0)
        out_label_ids = np.concatenate(out_label_ids, axis=0)
        gathered_logits_by_rel_prompt = np.concatenate(gathered_logits_by_rel_prompt, axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            'avg_loss': eval_loss,
        }

        preds = pred_probs.copy()
        preds = (preds > threshold).astype(int)

        prec, recall, f1, _ = compute_binary_f1(preds=preds, labels=out_label_ids)

        results.update({
            'precision': prec,
            'recall': recall,
            'f1': f1,
            'pred_probs': pred_probs,
            'predictions': preds,
            'labels': out_label_ids,
            'gathered_logits_by_rel_prompt': gathered_logits_by_rel_prompt, 
        })

        return results
    
##### #####

#### codes for optimizing similarity measurement #####

    def NLIBased_Optim_w_PosNeg_baseline(self, K=5, backup=50, div='test', neg_pos_ratio=1, ckpt_sub_folder='simi_optim_NLIBased_baseline_ckpt/'):
        """
        Iterate through div dataset relations, construct by sampling positive and negative examples from the div dataset. Then for each single target relation, optimize the NLIBased similarity measurement model and evaluate.
        Assume the negative samples are always given.
        Currently only use the cross entropy loss.
        Assume positive samples are from rel2Kshot_locIds while negative samples are sampled from rel2backup_locIds and the eval_locIds will serve as a consistent evaluation data.
        Currently we assume of a relation's intermediate result file is found, we will skip this relation and use this intermediate result as the approximation of the runned results (there might be difference because different rounds of random sampling of the variables like rel2Kshot_locIds, eval_locIds, rel2backup_locIds).

        
        Args:
            K: number of positive samples
            backup: number of backup samples (for sampling negative samples)    
            div: dataset div for sampling and evaluation (we assume all are conducted within the same div)
            neg_pos_ratio: number of negative samples over positive samples for each negative class to sample (the resulted number will be guaranteed to >= 1)
            ckpt_sub_folder: relative folder under self.args.cache_sub_dir to store the intermediate and final results

        Returns: a dict as
            {
                'div': div,
                'args': self.args,
                'rel_set': rel_set, list of relations of div dataset to iterate,
                'rel2Kshot_locIds': rel2Kshot_locIds, from relation to K shot local indices,
                'eval_locIds': eval_locIds, list of evaluation data's local indices,
                'rel2backup_locIds': rel2backup_locIds, from relation to backup local indices,
                'rel': list of relations iterated,
                'rel_def_prompt': list of rel_def_prompt for each relation,
                'rel_def_paraphrased_prompts': list of paraphrased relation def prompts for each relation,
                'neg_locIds': list of negative samples' local indices actually sampled,
                'rel_hist_evaluation_results': list of hist_evaluation_results corresponding to each relation,
            }

        """
        rel_set, rel2Kshot_locIds, eval_locIds, rel2backup_locIds = self.dataloader.sample_and_gather_indices(div=div, K=K, backup=backup)

        writer = SummaryWriter(log_dir=os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, 'tensorboard/'))


        hist_evaluation_results = {
            'div': div,
            'args': self.args,
            'rel_set': rel_set,
            'rel2Kshot_locIds': rel2Kshot_locIds,
            'eval_locIds': eval_locIds,
            'rel2backup_locIds': rel2backup_locIds,
            'rel': [],
            'rel_def_prompt': [],
            'rel_def_paraphrased_prompts': [],
            'neg_locIds': [],
            'rel_hist_evaluation_results': [],
        }


        for rel in rel_set:

            # initialize relation specific prompts
            rel_def_prompt = self.dataloader.rel_info[rel]["typed_desc_prompt"]
            rel_def_paraphrased_prompts = self.dataloader.rel_info[rel]['prompts']
            hist_evaluation_results['rel_def_prompt'].append(rel_def_prompt)
            hist_evaluation_results['rel_def_paraphrased_prompts'].append(rel_def_paraphrased_prompts)


            # use no paraphrase prompt
            rel_def_paraphrased_prompts = []


            # initialize model
            rel_NLI_model = NLIBasedSimilarityModel(args=self.args, tokenizer=self.dataloader.tokenizer, target_relation=rel, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts)

            # sampling negative data from backup data and use K shot positive data as positive training data
            div2local_indices = { # dict from div to list of local indices (both positive and negative samples)
                div: [],
            }

            div2assigned_labels = { # dict from div to list of single element tensors: torch.tensor([1], dtype=torch.long)/torch.tensor([0], dtype=torch.long)
                div: [],
            }

            test_div2local_indices = { # dict from div to list of local indices for test samples. Their ground truth labels will be used and are assigned by dataloader
                div: eval_locIds,
            }


            # gather positive samples
            div2local_indices[div].extend(rel2Kshot_locIds[rel])
            for sample_id in rel2Kshot_locIds[rel]:
                div2assigned_labels[div].append(torch.tensor([1], dtype=torch.long))

            num_neg_per_rel = max(1, int(neg_pos_ratio * K)) # number of negative samples for each relation

            # negative sampling
            for rel_neg in rel_set:
                # if we comment out the following two lines, we basically want to do random negative sampling over both negative classes and positive classes
                if rel_neg == rel: 
                    continue
                rel_neg_sample_ids = random.sample(rel2backup_locIds[rel_neg], num_neg_per_rel)
                hist_evaluation_results['neg_locIds'].append(rel_neg_sample_ids)
                div2local_indices[div].extend(rel_neg_sample_ids)
                for sample_id in rel_neg_sample_ids:
                    div2assigned_labels[div].append(torch.tensor([0], dtype=torch.long))
                
            # visualize
            print(f"    Relation definition prompt: {rel_def_prompt}")
            for i, rel_def_paraphrased_prompt in enumerate(rel_def_paraphrased_prompts):
                print(f"        Paraphrased prompt {i}:\t{rel_def_paraphrased_prompt}")
            print(f"\n===Train and evaluation on the relation {rel}===")
            print(f"    Number of positive training samples {len(rel2Kshot_locIds[rel])}")
            print(f"    Positive training sample ids: {rel2Kshot_locIds[rel]}")
            print(f"    Number of negative training samples: {len(div2local_indices[div]) - len(rel2Kshot_locIds[rel])}")
            print(f"    Number of evaluation samples: {len(eval_locIds)}")
            print(f"    Total training sample local indices: {div2local_indices[div]}")
            
            # training and test of the NLIBased 
            rel_hist_evaluation_results, ckpt_exists, _ = self.NLIBased_Optim_w_PosNeg_rel(rel_NLI_model=rel_NLI_model, target_rel=rel, rel_def_prompt=rel_def_prompt, rel_def_paraphrased_prompts=rel_def_paraphrased_prompts, div2local_indices=div2local_indices, div2assigned_labels=div2assigned_labels, test_div2local_indices=test_div2local_indices, ckpt_sub_folder=ckpt_sub_folder, SummaryWriter=writer)

            if ckpt_exists:
                print(f"===Ckpt found for relation: {rel}. Skipped its optimization round. Use the ckpt as an approximation instead.===")

            hist_evaluation_results['rel'].append(rel)
            hist_evaluation_results['rel_hist_evaluation_results'].append(rel_hist_evaluation_results)

        print(f"Averaged results for all relations: P: {np.mean([i['precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")

        print(f"Chosen results averaged over all relations: P: {np.mean([i['chosen_precision'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | R: {np.mean([i['chosen_recall'] for i in hist_evaluation_results['rel_hist_evaluation_results']])} | F1: {np.mean([i['chosen_f1'] for i in hist_evaluation_results['rel_hist_evaluation_results']])}")

        self.save_model(
            os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_all.pt'),
            hist_evaluation_results,
        )
        
        return hist_evaluation_results
        
    
    def NLIBased_Optim_w_PosNeg_rel(self, rel_NLI_model, target_rel, rel_def_prompt, rel_def_paraphrased_prompts, div2local_indices, div2assigned_labels, test_div2local_indices, ckpt_sub_folder='simi_optim_NLIBased_ckpt/', train_examples=None, train_assigned_labels=None, SummaryWriter=None, extra_info_to_save=None, dev_examples=None, dev_assigned_labels=None, dev_examples_1=None, dev_assigned_labels_1=None, dev_examples_2=None, dev_assigned_labels_2=None):
        """
        Based on the given positive and negative examples for a single target relation, optimize the NLIBased similarity measurement model and evaluate.
        Assume the negative samples are always given.
        Based on the training instant loss curve, pick the epoch at the elbow point and return the corresponding model ckpt for further inference usage (deleting all the other epochs' model ckpts).

        Currently only use the BCE loss. 

        Args:
            rel_NLI_model: NLIBased model for a specific relation
            target_rel: target relation
            div2local_indices: dict from div to local indices (list[int]) referring to both positive and negative examples
            div2assigned_labels: dict from div to a python list of one dimenisonal tensors with dtype=torch.long (e.g., such tensor can be constructed by "torch.tensor([1], dtype=torch.long)")
            test_div2local_indices: dict from div to local indices (list[int]) referring to evaluation data (the labels will be derived by dataloader's function, using the ground truth labels)
            SummaryWriter: tensorboard summary writer

        Returns:
            rel_hist_evaluation_results: dict as
                {
                    'rel': the target relation,
                    'rel_def_prompt': rel_def_prompt, relation definition prompt,
                    'rel_def_paraphrased_prompts': rel_def_paraphrased_prompts, paraphrased relation prompts,
                    'last_evaluation_results': last evaluation result, a dict,
                    'rel_NLI_model_state_dict': model ckpt,
                    'precision': list of history evaluated precision,
                    'recall': list of history evaluated recall,
                    'f1': list of history evaluated f1,
                    'eval_epochs': list of epoch index  that precision, recall, f1 is calculated (so this list could have repetitions)
                    'test_examples': list of test examples (list of dicts) retrieved by dataloader,
                    'chosen_rel_NLI_model_state_dict': the model ckpt corresponding to the elbow epoch of the instant training loss curve # Only updated at the end of this function and won't be saved in intermediate saving of rel_hist_evaluation_results
                    'chosen_rel_NLI_model_ckpt_epoch': the elbow epoch of the instant training loss curve  # Only updated at the end of this function and won't be saved in intermediate saving of rel_hist_evaluation_results
                    'chosen_precision': evaluated precision on test examples at the 'chosen_rel_NLI_model_ckpt_epoch'
                    'chosen_recall': evaluated recall on test examples at the 'chosen_rel_NLI_model_ckpt_epoch'
                    'chosen_f1': evaluated f1 on test examples at the 'chosen_rel_NLI_model_ckpt_epoch'
                }
        """

        def get_NLIBased_Optim_w_PosNeg_optimizer(model, dataloader):
            """
            Return the optimizer and scheduler and total steps for optimizing the model on the data.
            """
            if self.args.max_steps > 0:
                t_total = self.args.max_steps
                self.args.num_train_epochs = self.args.max_steps // (len(dataloader) // self.args.accum_steps) + 1
            else:
                t_total = len(dataloader) // self.args.accum_steps * self.args.num_train_epochs

            cur_model = model.module if hasattr(model, 'module') else model

            # prepare optimizer and scheduler (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in cur_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in cur_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]


            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

            return optimizer, scheduler, t_total
    
        def NLIBased_Optim_evaluate(model, eval_dataloader, threshold=0.5):
            """
            Evaluate the model, caculate the prediction and output the evaluation results. 

            Args:
                model: model to be evaluated, on GPU (if possible)
                eval_dataloader: evaluation dataloader
                threshold: logits > than this threshold will lead to a positive prediction label
            Returns: 
                Results: a dict
                    {
                        'precision': precision score,
                        'recall': recall score,
                        'f1': f1 score,
                        'pred_probs': predicted probability distribution, # numpy array of (#test set, 2),
                        'predictions': predictions after applying the threshold, # numpy array of (#test set, ),
                        'labels': gold labels for binary classification, # numpy array of (#test set, ),
                        'gathered_pred_probs_by_rel_prompt': model output logits by prompt by original entailment classes, #  numpy array of (#test set, len(rel_prompts), 3),
                    }

            """
            # test the model
            print(f'   Start evaluating the model...')
            print(f'   Total test batch size: {self.args.eval_batch_size}')

            eval_loss = 0.0
            nb_eval_steps = 0
            pred_probs = []
            out_label_ids = []
            gathered_pred_probs_by_rel_prompt = []

            binary_classification_criterion = nn.CrossEntropyLoss(reduction='mean')
            binary_classification_criterion_alternative = nn.BCELoss(reduction='sum')

            model.eval()
            for batch in tqdm(eval_dataloader, desc='Evaluating'):
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'assigned_labels': batch[2],
                    }

                    logits_by_prompt, _, pred_logits = model(**inputs)


                    # loss = binary_classification_criterion(pred_logits, batch[2])
                    loss = binary_classification_criterion_alternative(pred_logits[:, 1], batch[2].float())
                
                    eval_loss += loss.item()

                nb_eval_steps += 1

                pred_probs.append(pred_logits.detach().cpu().numpy())
                out_label_ids.append(inputs['assigned_labels'].detach().cpu().numpy())
                gathered_pred_probs_by_rel_prompt.append(logits_by_prompt.detach().cpu().numpy())
            
            pred_probs = np.concatenate(pred_probs, axis=0)
            out_label_ids = np.concatenate(out_label_ids, axis=0)
            gathered_pred_probs_by_rel_prompt = np.concatenate(gathered_pred_probs_by_rel_prompt, axis=0)

            # eval_loss = eval_loss / nb_eval_steps
            results = {
                # 'avg_loss': eval_loss / nb_eval_steps,
                'avg_eval_loss': eval_loss / len(eval_dataloader),
                'threshold': threshold,
            }

            preds = pred_probs.copy()
            preds = (preds[:, 1] > threshold).astype(int)

            prec, recall, f1, _ = compute_binary_f1(preds=preds, labels=out_label_ids)

            results.update({
                'eval_loss': eval_loss,
                'nb_eval_steps': nb_eval_steps,
                'precision': prec,
                'recall': recall,
                'f1': f1,
                'pred_probs': pred_probs, # numpy array of (#test set, 2)
                'predictions': preds, # numpy array of (#test set, )
                'labels': out_label_ids, # numpy array of (#test set, )
                'gathered_pred_probs_by_rel_prompt': gathered_pred_probs_by_rel_prompt, #  numpy array of (#test set, len(rel_prompts), 3)
            })

            return results
        

        # in case the ckpt_file exists and the model is not loaded to gpu
        print(f"\n\n\n===Conducting similarity model optimization on relation {target_rel} ({self.dataloader.rel_info[target_rel]['relation_name']})")


        ckpt_file = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_{target_rel}.pt')


        if os.path.exists(ckpt_file):
            rel_hist_evaluation_results = torch.load(ckpt_file)
            return rel_hist_evaluation_results, True, rel_NLI_model # True indicates the ckpt already exists

        # for visualization cache
        rel_visual_cache_folder =  os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'visual_cache/', f'{target_rel}/')
        os.makedirs(rel_visual_cache_folder, exist_ok=True)


        if torch.cuda.device_count() > 1:
            print('Wrapping model with torch.nn.DataParallel...')
            rel_NLI_model = torch.nn.DataParallel(rel_NLI_model)
        rel_NLI_model.to(self.device)

        # for saving purposes
        pos_examples_to_save, neg_examples_to_save = [], []


        if train_examples is None:
            train_locIds, train_divs, train_labels = [], [], []
            # train_locIds: list of local indices corresponding to the div 
            # train_divs: list of dataset divs corresponding to each local index in train_locIds
            # train_labels: list of one dimenisonal tensors with dtype=torch.long, indicating the example is positive or negative

            label2relative_ids = { # dict from label category to relative indices (indexing train_locIds, train_divs, train_labels) for corresponding examples
                'pos': [], 'neg':[],
            }
            for div, local_indices in div2local_indices.items():
                assigned_labels = div2assigned_labels[div]
                for i, local_idx in enumerate(local_indices):
                    train_locIds.append(local_idx)
                    train_divs.append(div)
                    train_labels.append(assigned_labels[i])

                    if assigned_labels[i] == 1:
                        label2relative_ids['pos'].append(len(train_locIds) - 1)
                        pos_examples_to_save.extend(self.dataloader.get_div_raw_data_w_indices(div=div, div_indices=[local_idx]))
                    else:
                        label2relative_ids['neg'].append(len(train_locIds) - 1)
                        neg_examples_to_save.extend(self.dataloader.get_div_raw_data_w_indices(div=div, div_indices=[local_idx]))

            train_dataloader = self.dataloader.make_dataloader_runtime_NLIBased(div2local_indices=div2local_indices, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts, batch_size=self.args.train_batch_size, div2assigned_labels=div2assigned_labels, random_sample=True, target_rel=target_rel, ckpt_sub_folder=ckpt_sub_folder, ckpt_file_suffix='_train')
        else: # training examples are given 
            train_dataloader = self.dataloader.make_dataloader_runtime_NLIBased(div2local_indices=None, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts, batch_size=self.args.train_batch_size, div2assigned_labels=None, random_sample=True, target_rel=target_rel, ckpt_sub_folder=ckpt_sub_folder, ckpt_file_suffix='_train', train_examples=train_examples, train_assigned_labels=train_assigned_labels)

            for train_example, train_assigned_label in zip(train_examples, train_assigned_labels):
                if train_assigned_label == 1: pos_examples_to_save.append(train_example)
                else: neg_examples_to_save.append(train_example)


        dev_dataloader = self.dataloader.make_dataloader_runtime_NLIBased(div2local_indices=None, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts, batch_size=self.args.eval_batch_size, div2assigned_labels=None, random_sample=False, target_rel=target_rel, ckpt_sub_folder=ckpt_sub_folder, ckpt_file_suffix='_dev', train_examples=dev_examples, train_assigned_labels=dev_assigned_labels)
        
        if dev_examples_1 is not None: # requires the dev_examples_2 and the corresponding dev_assigned_labels_1, dev_assigned_labels_2 should not be None
            dev_dataloader_1 = self.dataloader.make_dataloader_runtime_NLIBased(div2local_indices=None, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts, batch_size=self.args.eval_batch_size, div2assigned_labels=None, random_sample=False, target_rel=target_rel, ckpt_sub_folder=ckpt_sub_folder, ckpt_file_suffix='_dev_1', train_examples=dev_examples_1, train_assigned_labels=dev_assigned_labels_1)
            
            dev_dataloader_2 = self.dataloader.make_dataloader_runtime_NLIBased(div2local_indices=None, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts, batch_size=self.args.eval_batch_size, div2assigned_labels=None, random_sample=False, target_rel=target_rel, ckpt_sub_folder=ckpt_sub_folder, ckpt_file_suffix='_dev_2', train_examples=dev_examples_2, train_assigned_labels=dev_assigned_labels_2)
        
        
        test_dataloader = self.dataloader.make_dataloader_runtime_NLIBased(div2local_indices=test_div2local_indices, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts, batch_size=self.args.eval_batch_size, div2assigned_labels=None, random_sample=False, target_rel=target_rel, ckpt_sub_folder=ckpt_sub_folder, ckpt_file_suffix='_test') # set div2assigned_labels to None so that the ground truth label can be assigned by dataloader


        optimizer, scheduler, t_total = get_NLIBased_Optim_w_PosNeg_optimizer(model=rel_NLI_model, dataloader=train_dataloader)


        # train the model 
        print(f'Start training the model...')
        print(f'|\tNum Epochs: {self.args.num_train_epochs}')
        print(f'|\tTotal effective train batch size: {self.args.train_batch_size * self.args.accum_steps}')
        print(f'|\tGradient accumulation steps: {self.args.accum_steps}')
        print(f'|\tTotal optimization steps: {t_total}')
        

        global_step = 0 # an effective batch will count as 1 global_step
        tr_loss = 0.0

        # the following two lists contain the instant training loss (averaged over batch, not effective batch) and the corresponding epoch at each step of optimizer.step() 
        instant_tr_loss_list = []
        instant_tr_loss_epoch_list = []
        # will save model ckpt at each epoch under following folder, at the end of this function, pick one ckpt to return and will delete all the other ckpts under this folder
        epoch_model_ckpt_folder = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{target_rel}/epoch_model_ckpt/')
        os.makedirs(epoch_model_ckpt_folder, exist_ok=True)

        last_evaluation_results = None
        last_dev_evaluation_results = None
        
        # history evaluation results during training for this target_relation
        rel_hist_evaluation_results = {
            'rel': target_rel,
            'rel_def_prompt': rel_def_prompt,
            'rel_def_paraphrased_prompts': rel_def_paraphrased_prompts,
            'last_evaluation_results': None,
            'precision': [],
            'recall': [],
            'f1': [],
            'eval_epochs': [],
            'test_examples': self.dataloader.get_div_raw_data_w_indices(div=list(test_div2local_indices.keys())[0], div_indices=test_div2local_indices[list(test_div2local_indices.keys())[0]]),
            'chosen_rel_NLI_model_ckpt_epoch': None,
            'chosen_rel_NLI_model_state_dict_path': None,
            'chosen_precision': None,
            'chosen_recall': None,
            'chosen_f1': None,
            'elbow_epoch': None,
            'pos_examples': pos_examples_to_save,
            'neg_examples': neg_examples_to_save,
            'saved_evaluation_results': {},
            'dev_precision': [],
            'dev_recall': [],
            'dev_f1': [],
            'dev_eval_epochs': [],
            'dev_chosen_rel_NLI_model_ckpt_epoch': None,
            'dev_chosen_rel_NLI_model_state_dict_path': None,
            'dev_chosen_precision': None,
            'dev_chosen_recall': None,
            'dev_chosen_f1': None,
            'dev_1_precision': [],
            'dev_1_recall': [],
            'dev_1_f1': [],
            'dev_1_eval_epochs': [],
            'dev_1_chosen_rel_NLI_model_ckpt_epoch': None,
            'dev_1_chosen_rel_NLI_model_state_dict_path': None,
            'dev_1_chosen_precision': None,
            'dev_1_chosen_recall': None,
            'dev_1_chosen_f1': None,
            'dev_2_precision': [],
            'dev_2_recall': [],
            'dev_2_f1': [],
            'dev_2_eval_epochs': [],
            'dev_2_chosen_rel_NLI_model_ckpt_epoch': None,
            'dev_2_chosen_rel_NLI_model_state_dict_path': None,
            'dev_2_chosen_precision': None,
            'dev_2_chosen_recall': None,
            'dev_2_chosen_f1': None,
        }

        if extra_info_to_save is not None: rel_hist_evaluation_results.update(extra_info_to_save)

        rel_NLI_model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc='Epoch')
        margin_criterion = torch.nn.MarginRankingLoss(margin=0.2).to(self.device)
        binary_classification_criterion = nn.CrossEntropyLoss(reduction='mean')
        binary_classification_criterion_alternative = nn.BCELoss(reduction='mean')

        
        for epoch_i in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                rel_NLI_model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'assigned_labels': batch[2],
                }

                logits_by_prompt, _, pred_logits = rel_NLI_model(**inputs)

                # loss = binary_classification_criterion(pred_logits, batch[2])
                loss = binary_classification_criterion_alternative(pred_logits[:, 1], batch[2].float())
                
                if self.args.accum_steps > 1:
                    loss = loss / self.args.accum_steps
                    
                loss.backward()
                tr_loss += loss.item()


                if (step + 1) % self.args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(rel_NLI_model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    rel_NLI_model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description(f'iteration: {step}, loss: {tr_loss / global_step}')

                    if SummaryWriter is not None:
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/avg_tr_loss", tr_loss / global_step, global_step)
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/instant_tr_loss", loss.item() * self.args.accum_steps, global_step)


                    instant_tr_loss_list.append(loss.item() * self.args.accum_steps)
                    instant_tr_loss_epoch_list.append(epoch_i)


                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        last_evaluation_results = NLIBased_Optim_evaluate(model=rel_NLI_model, eval_dataloader=test_dataloader, threshold=0.5) # evaluation function
                        rel_hist_evaluation_results['last_evaluation_results'] = last_evaluation_results
                        rel_hist_evaluation_results['precision'].append(last_evaluation_results['precision'])
                        rel_hist_evaluation_results['recall'].append(last_evaluation_results['recall'])
                        rel_hist_evaluation_results['f1'].append(last_evaluation_results['f1'])
                        rel_hist_evaluation_results['eval_epochs'].append(epoch_i)


                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if isinstance(rel_NLI_model, nn.DataParallel):
                            ckpt_model_state_dict = rel_NLI_model.module.state_dict()
                        else:
                            ckpt_model_state_dict = rel_NLI_model.state_dict()
                        

                        rel_hist_evaluation_results.update({
                            'epoch': epoch_i,
                            'step': step,
                            'args': self.args,
                        })
                        self.save_model(
                            os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_{target_rel}_intermediate.pt'),
                            rel_hist_evaluation_results,
                        )

                        self.save_model(
                            save_path=os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{epoch_i}.pt'),
                            pytorch_save_dict=ckpt_model_state_dict,
                        )

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            
            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            if self.args.logging_steps == 0: # logging at the end of every epoch
                last_evaluation_results = NLIBased_Optim_evaluate(model=rel_NLI_model, eval_dataloader=test_dataloader, threshold=0.5) # evaluation function
                rel_hist_evaluation_results['last_evaluation_results'] = last_evaluation_results
                rel_hist_evaluation_results['precision'].append(last_evaluation_results['precision'])
                rel_hist_evaluation_results['recall'].append(last_evaluation_results['recall'])
                rel_hist_evaluation_results['f1'].append(last_evaluation_results['f1'])
                rel_hist_evaluation_results['eval_epochs'].append(epoch_i)
                
                
                last_dev_evaluation_results = NLIBased_Optim_evaluate(model=rel_NLI_model, eval_dataloader=dev_dataloader, threshold=0.5) 
                rel_hist_evaluation_results['dev_precision'].append(last_dev_evaluation_results['precision'])
                rel_hist_evaluation_results['dev_recall'].append(last_dev_evaluation_results['recall'])
                rel_hist_evaluation_results['dev_f1'].append(last_dev_evaluation_results['f1'])
                rel_hist_evaluation_results['dev_eval_epochs'].append(epoch_i)
                
                if dev_examples_1 is not None:
                    last_dev_evaluation_results = NLIBased_Optim_evaluate(model=rel_NLI_model, eval_dataloader=dev_dataloader_1, threshold=0.5) 
                    rel_hist_evaluation_results['dev_1_precision'].append(last_dev_evaluation_results['precision'])
                    rel_hist_evaluation_results['dev_1_recall'].append(last_dev_evaluation_results['recall'])
                    rel_hist_evaluation_results['dev_1_f1'].append(last_dev_evaluation_results['f1'])
                    rel_hist_evaluation_results['dev_1_eval_epochs'].append(epoch_i)
                    
                    last_dev_evaluation_results = NLIBased_Optim_evaluate(model=rel_NLI_model, eval_dataloader=dev_dataloader_2, threshold=0.5) 
                    rel_hist_evaluation_results['dev_2_precision'].append(last_dev_evaluation_results['precision'])
                    rel_hist_evaluation_results['dev_2_recall'].append(last_dev_evaluation_results['recall'])
                    rel_hist_evaluation_results['dev_2_f1'].append(last_dev_evaluation_results['f1'])
                    rel_hist_evaluation_results['dev_2_eval_epochs'].append(epoch_i)
                
                
                if epoch_i in self.args.logging_epochs:
                    rel_hist_evaluation_results['saved_evaluation_results'][epoch_i] = last_evaluation_results
                
                if SummaryWriter is not None:
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/precision", last_evaluation_results['precision'], epoch_i)
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/recall", last_evaluation_results['recall'], epoch_i)
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/f1", last_evaluation_results['f1'], epoch_i)
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/eval_loss", last_evaluation_results['eval_loss'], epoch_i)
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/avg_eval_loss", last_evaluation_results['avg_eval_loss'], epoch_i)

                    
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_precision", rel_hist_evaluation_results['dev_precision'][-1], epoch_i)
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_recall", rel_hist_evaluation_results['dev_recall'][-1], epoch_i)
                    SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_f1", rel_hist_evaluation_results['dev_f1'][-1], epoch_i)
                    
                    if dev_examples_1 is not None:
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_1_precision", rel_hist_evaluation_results['dev_1_precision'][-1], epoch_i)
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_1_recall", rel_hist_evaluation_results['dev_1_recall'][-1], epoch_i)
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_1_f1", rel_hist_evaluation_results['dev_1_f1'][-1], epoch_i)
                        
                        
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_2_precision", rel_hist_evaluation_results['dev_2_precision'][-1], epoch_i)
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_2_recall", rel_hist_evaluation_results['dev_2_recall'][-1], epoch_i)
                        SummaryWriter.add_scalar(f"{target_rel}({self.dataloader.rel_info[target_rel]['relation_name']})/dev_2_f1", rel_hist_evaluation_results['dev_2_f1'][-1], epoch_i)
                    
                    
                plot_prediction_distributions(test_labels=last_evaluation_results['labels'], predictions=last_evaluation_results['pred_probs'], bins=100, save_path=os.path.join(rel_visual_cache_folder, f'{target_rel}_{epoch_i}.png'))
                
                train_iterator.set_description(f"Relation: {target_rel}: P={last_evaluation_results['precision']}, R={last_evaluation_results['recall']}, F={last_evaluation_results['f1']}, Epoch")

            if self.args.save_steps == 0:
                if isinstance(rel_NLI_model, nn.DataParallel):
                    ckpt_model_state_dict = rel_NLI_model.module.state_dict()
                else:
                    ckpt_model_state_dict = rel_NLI_model.state_dict()

                rel_hist_evaluation_results.update({
                    'epoch': epoch_i,
                    'step': step,
                    'args': self.args,
                })

                self.save_model(
                    os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'result_{target_rel}_intermediate.pt'),
                    rel_hist_evaluation_results,
                )

                self.save_model(
                    save_path=os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{epoch_i}.pt'),
                    pytorch_save_dict=ckpt_model_state_dict,
                )
            else: # for experimental purpose, we need to pick up a ckpt in the end for snowballing, so we need to save ckpts anyway
                if isinstance(rel_NLI_model, nn.DataParallel):
                    ckpt_model_state_dict = rel_NLI_model.module.state_dict()
                else:
                    ckpt_model_state_dict = rel_NLI_model.state_dict()
                    
                self.save_model(
                    save_path=os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{epoch_i}.pt'),
                    pytorch_save_dict=ckpt_model_state_dict,
                )

        # here we override the hyperparameter and assumes that for self.args.logging_steps < 0, we will also log at the end
        if self.args.logging_steps < 0: # logging at the end of training
            last_evaluation_results = NLIBased_Optim_evaluate(model=rel_NLI_model, eval_dataloader=test_dataloader, threshold=0.5) # evaluation function
            rel_hist_evaluation_results['last_evaluation_results'] = last_evaluation_results
            rel_hist_evaluation_results['precision'].append(last_evaluation_results['precision'])
            rel_hist_evaluation_results['recall'].append(last_evaluation_results['recall'])
            rel_hist_evaluation_results['f1'].append(last_evaluation_results['f1'])
            rel_hist_evaluation_results['eval_epochs'].append(epoch_i)
            train_iterator.set_description(f"Relation: {target_rel}: P={last_evaluation_results['precision']}, R={last_evaluation_results['recall']}, F={last_evaluation_results['f1']}, Epoch")


        print(f"Averaged results for {target_rel}:  P: {np.mean(rel_hist_evaluation_results['precision'])} | R: {np.mean(rel_hist_evaluation_results['recall'])} | F1: {np.mean(rel_hist_evaluation_results['f1'])}")
        print(f"Latest results for {target_rel}:  P: {rel_hist_evaluation_results['precision'][-1]} | R: {rel_hist_evaluation_results['recall'][-1]} | F1: {rel_hist_evaluation_results['f1'][-1]}")


        # here we override the hyperparameter and assumes that for self.args.save_steps < 0, we will also save at the end
        if self.args.save_steps < 0:
            if isinstance(rel_NLI_model, nn.DataParallel):
                ckpt_model_state_dict = rel_NLI_model.module.state_dict()
            else:
                ckpt_model_state_dict = rel_NLI_model.state_dict()


            rel_hist_evaluation_results.update({
                'epoch': int(self.args.num_train_epochs) - 1,
                'step': step,
                'args': self.args,
            })

            self.save_model(
                ckpt_file,
                rel_hist_evaluation_results,
            )
        
            self.save_model(
                save_path=os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{int(self.args.num_train_epochs) - 1}.pt'),
                pytorch_save_dict=ckpt_model_state_dict,
            )


        # select which ckpt we want to pick for later snowballing (or inference over unlabeled corpus)
        try:
            print(f"Starting to find elbow points with x_values(instant_tr_loss_epoch_list)={instant_tr_loss_epoch_list}, y_values(instant_tr_loss_list)={instant_tr_loss_list}")
            chosen_model_ckpt_epoch, _ = get_elbow_x_value(y_values=instant_tr_loss_list, x_values=instant_tr_loss_epoch_list, y_threshold=0.3, S=1.0, curve='convex', direction='decreasing', online=False)
            print(f"Found elbow chosen_model_ckpt_epoch={chosen_model_ckpt_epoch}")
        except Exception as e:
            print("Error occurred during find the elbow epoch: ", e)
            chosen_model_ckpt_epoch = 0

        rel_hist_evaluation_results['elbow_epoch'] = chosen_model_ckpt_epoch


        if not self.args.choose_by_elbow:
            final_chosen_model_ckpt_epoch = self.args.save_epochs[-1]
        else:
            final_chosen_model_ckpt_epoch = chosen_model_ckpt_epoch
        print(f"Since self.args.choose_by_elbow={self.args.choose_by_elbow}, we take final_chosen_model_ckpt_epoch={final_chosen_model_ckpt_epoch}")
        important_epochs_to_save = deepcopy(self.args.save_epochs)
        important_epochs_to_save.append(chosen_model_ckpt_epoch)

        
        dev_argmax_index = np.argmax(rel_hist_evaluation_results['dev_f1'])
        dev_chosen_model_ckpt_epoch = rel_hist_evaluation_results['dev_eval_epochs'][dev_argmax_index]
        rel_hist_evaluation_results['dev_chosen_rel_NLI_model_state_dict_path'] = os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{dev_chosen_model_ckpt_epoch}.pt')
        rel_hist_evaluation_results['dev_chosen_rel_NLI_model_ckpt_epoch'] = dev_chosen_model_ckpt_epoch
        important_epochs_to_save.append(dev_chosen_model_ckpt_epoch)
        rel_hist_evaluation_results['dev_chosen_precision'] = rel_hist_evaluation_results['precision'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
        rel_hist_evaluation_results['dev_chosen_recall'] = rel_hist_evaluation_results['recall'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
        rel_hist_evaluation_results['dev_chosen_f1'] = rel_hist_evaluation_results['f1'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
        
        if dev_examples_1 is not None:
            dev_argmax_index = np.argmax(rel_hist_evaluation_results['dev_1_f1'])
            dev_chosen_model_ckpt_epoch = rel_hist_evaluation_results['dev_1_eval_epochs'][dev_argmax_index]
            rel_hist_evaluation_results['dev_1_chosen_rel_NLI_model_state_dict_path'] = os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{dev_chosen_model_ckpt_epoch}.pt')
            rel_hist_evaluation_results['dev_1_chosen_rel_NLI_model_ckpt_epoch'] = dev_chosen_model_ckpt_epoch
            important_epochs_to_save.append(dev_chosen_model_ckpt_epoch)
            rel_hist_evaluation_results['dev_1_chosen_precision'] = rel_hist_evaluation_results['precision'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
            rel_hist_evaluation_results['dev_1_chosen_recall'] = rel_hist_evaluation_results['recall'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
            rel_hist_evaluation_results['dev_1_chosen_f1'] = rel_hist_evaluation_results['f1'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
            
            dev_argmax_index = np.argmax(rel_hist_evaluation_results['dev_2_f1'])
            dev_chosen_model_ckpt_epoch = rel_hist_evaluation_results['dev_2_eval_epochs'][dev_argmax_index]
            rel_hist_evaluation_results['dev_2_chosen_rel_NLI_model_state_dict_path'] = os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{dev_chosen_model_ckpt_epoch}.pt')
            rel_hist_evaluation_results['dev_2_chosen_rel_NLI_model_ckpt_epoch'] = dev_chosen_model_ckpt_epoch
            important_epochs_to_save.append(dev_chosen_model_ckpt_epoch)
            rel_hist_evaluation_results['dev_2_chosen_precision'] = rel_hist_evaluation_results['precision'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
            rel_hist_evaluation_results['dev_2_chosen_recall'] = rel_hist_evaluation_results['recall'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]
            rel_hist_evaluation_results['dev_2_chosen_f1'] = rel_hist_evaluation_results['f1'][rel_hist_evaluation_results['eval_epochs'].index(dev_chosen_model_ckpt_epoch)]

        model_ckpt_file_pattern = re.compile(r'model_ckpt_epoch_(\d+)\.pt')
        for filename in os.listdir(epoch_model_ckpt_folder):
            match = model_ckpt_file_pattern.match(filename)
            if match:
                temp_epoch_i = int(match.group(1))
                # if temp_epoch_i != chosen_model_ckpt_epoch:
                if temp_epoch_i not in important_epochs_to_save:
                    print(f"Removing ckpt filename={filename}")
                    os.remove(os.path.join(epoch_model_ckpt_folder, filename))
        
        rel_hist_evaluation_results['chosen_rel_NLI_model_state_dict_path'] = os.path.join(epoch_model_ckpt_folder, f'model_ckpt_epoch_{final_chosen_model_ckpt_epoch}.pt')
        rel_hist_evaluation_results['chosen_rel_NLI_model_ckpt_epoch'] = final_chosen_model_ckpt_epoch


        try:
            i = rel_hist_evaluation_results['eval_epochs'].index(final_chosen_model_ckpt_epoch)
        except ValueError:
            print(f"Error: no evaluations results on the final_chosen_rel_NLI_model_ckpt_epoch (={final_chosen_model_ckpt_epoch}) where eval_epochs={rel_hist_evaluation_results['eval_epochs']}, so we take the last epoch results as default")
            i = -1
        rel_hist_evaluation_results['chosen_precision'] = rel_hist_evaluation_results['precision'][i]
        rel_hist_evaluation_results['chosen_recall'] = rel_hist_evaluation_results['recall'][i]
        rel_hist_evaluation_results['chosen_f1'] = rel_hist_evaluation_results['f1'][i]

        self.save_model(
            ckpt_file,
            rel_hist_evaluation_results,
        )
        
        print(f"Chosen results for {target_rel}:  P: {rel_hist_evaluation_results['chosen_precision']} | R: {rel_hist_evaluation_results['chosen_recall']} | F1: {rel_hist_evaluation_results['chosen_f1']}")

        try:
            i = rel_hist_evaluation_results['eval_epochs'].index(chosen_model_ckpt_epoch)
        except ValueError:
            print(f"Error: no evaluations results on the elbow method chosen_rel_NLI_model_ckpt_epoch (={chosen_model_ckpt_epoch}) where eval_epochs={rel_hist_evaluation_results['eval_epochs']}, so we take the last epoch results as default")
            i = -1
        print(f"Elbow epoch results for {target_rel}:  P: {rel_hist_evaluation_results['precision'][i]} | R: {rel_hist_evaluation_results['recall'][i]} | F1: {rel_hist_evaluation_results['f1'][i]}")

        
        if torch.cuda.device_count() > 1:
            rel_NLI_model = rel_NLI_model.module.to('cpu')
        # del optimizer, scheduler, ckpt_model_state_dict, inputs, batch, loss

        gc.collect()
        torch.cuda.empty_cache()
        # rel_NLI_model.to(torch.device('cpu'))

        return rel_hist_evaluation_results, False, rel_NLI_model
    

##### #####




#### codes for snowball similarity measurement #####

def NLIBased_Inference(rank, world_size, args, model_path, dataloader, inference_examples, target_rel, rel_def_prompt, rel_def_paraphrased_prompts, shared_results, ckpt_sub_folder='simi_optim_NLIBased_ckpt/', threshold=0.5):
    """
    Based on the given rel_NLI_model and the examples for a single target relation, conduct inference on the examples and return the prediction. 
    This function is written for the set up of pytorch distributed data parallel inference.
    Args: 

    Returns:


    """
    
    # setup for ddp
    ddp_setup(rank, world_size, master_port=args.dist_port)

    if rank == 0:
        print(f'   Start inference on the unlabled corpus...')
        print(f'   Total inference batch size: {args.unlabel_infer_batch_size}')
    model = NLIBasedSimilarityModel(args=args, tokenizer=dataloader.tokenizer, target_relation=target_rel, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts)
    model.to(rank)
    
    
    update_model_state_dict(model=model, target_state_dict=torch.load(model_path, map_location='cuda'))
    

    model = DDP(model, device_ids=[rank])


    # construct the binary labels for unlabeled samples
    inference_binary_noisy_labels = [torch.tensor([int(inference_ex['relation'] == target_rel)], dtype=torch.long) for inference_ex in inference_examples]
    # get the dataloader
    eval_dataloader = dataloader.make_dataloader_runtime_NLIBased(div2local_indices=None, rel_prompts=[rel_def_prompt] + rel_def_paraphrased_prompts, batch_size=args.unlabel_infer_batch_size_per_device, div2assigned_labels=None, random_sample=False, target_rel=target_rel, ckpt_sub_folder=ckpt_sub_folder, ckpt_file_suffix=f'_unlabeled', train_examples=inference_examples, train_assigned_labels=inference_binary_noisy_labels, rank=rank)


    eval_loss = 0.0
    nb_eval_steps = 0
    pred_probs = []
    out_label_ids = []
    gathered_pred_probs_by_rel_prompt = []
    inference_relative_ids = []

    binary_classification_criterion_alternative = nn.BCELoss(reduction='sum')

    model.eval()
    # eval_dataloader.sampler.set_epoch(epoch)
    if rank == 0: eval_dataloader = tqdm(eval_dataloader, desc='Predicting')
    for batch in eval_dataloader:
        relative_ids = batch[3].numpy()
        batch = tuple(t.to(rank) for t in batch[:3])
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'assigned_labels': batch[2],
            }

            logits_by_prompt, _, pred_logits = model(**inputs)

            loss = binary_classification_criterion_alternative(pred_logits[:, 1], batch[2].float())

            eval_loss += loss.item()
        
        nb_eval_steps += 1

        pred_probs.append(pred_logits.detach().cpu().numpy())
        out_label_ids.append(inputs['assigned_labels'].detach().cpu().numpy())
        gathered_pred_probs_by_rel_prompt.append(logits_by_prompt.detach().cpu().numpy())
        inference_relative_ids.append(relative_ids)

    pred_probs = np.concatenate(pred_probs, axis=0)
    out_label_ids = np.concatenate(out_label_ids, axis=0)
    gathered_pred_probs_by_rel_prompt = np.concatenate(gathered_pred_probs_by_rel_prompt, axis=0)
    inference_relative_ids = np.concatenate(inference_relative_ids, axis=0)

    results = {
        # 'avg_eval_loss': eval_loss / len(eval_dataloader),
        'threshold': threshold,
    }
    
    preds = pred_probs.copy()
    preds = (preds[:, 1] > threshold).astype(int)

    # prec, recall, f1, _ = compute_binary_f1(preds=preds, labels=out_label_ids)

    results.update({
        'eval_loss': eval_loss,
        # 'nb_eval_steps': nb_eval_steps,
        # 'precision': prec,
        # 'recall': recall,
        # 'f1': f1,
        'pred_probs': pred_probs, # numpy array of (#unlabeled examples, 2)
        'predictions': preds, # numpy array of (#unlabeled examples, ) 
        'labels': out_label_ids, # numpy array of (#unlabeled examples, )
        'gathered_pred_probs_by_rel_prompt': gathered_pred_probs_by_rel_prompt, #  numpy array of (#unlabeled examples, len(rel_prompts), 3)
        'inference_relative_ids': inference_relative_ids, # numpy array of (#unlabeled examples, ) 
    })

    
    # gather results from all processes
    print(f'Gathering results on rank={rank}...')
    gathered_results = [None] * world_size  # placeholder for all results
    dist.all_gather_object(gathered_results, results)
    for i in gathered_results: assert i is not None
    print(f'Gathering done with rank={rank}')

    # put the gathered results into the shared results
    if rank == 0:  # only the main process puts the results in the queue
        gathered_results_sorted = {
            'threshold': [],
            'eval_loss': [],
            'pred_probs': [],
            'predictions': [],
            'labels': [],
            'gathered_pred_probs_by_rel_prompt': [],
            'inference_relative_ids': [],
        }
        append_and_sort_keys = ['pred_probs', 'predictions', 'labels', 'gathered_pred_probs_by_rel_prompt', 'inference_relative_ids']
        sum_keys = ['eval_loss', ]
        unchanged_keys = ['threshold', ]
        for i, result_dict in enumerate(gathered_results):
            for key in append_and_sort_keys:
                if i == 0: gathered_results_sorted[key] = result_dict[key]
                else: gathered_results_sorted[key] = np.concatenate((gathered_results_sorted[key], result_dict[key]), axis=0)
            for key in sum_keys:
                if i == 0: gathered_results_sorted[key] = result_dict[key]
                else: gathered_results_sorted[key] += result_dict[key]
            for key in unchanged_keys:
                if i == 0: gathered_results_sorted[key] = result_dict[key]

        sorted_indices = np.argsort(gathered_results_sorted['inference_relative_ids'])
        for key in append_and_sort_keys: gathered_results_sorted[key] = gathered_results_sorted[key][sorted_indices]

        torch.save(gathered_results_sorted, os.path.join(dataloader.args.dataset_dir, dataloader.args.cache_sub_dir, ckpt_sub_folder, f'{target_rel}_unlabeled_inference.pt'))
        shared_results.append(gathered_results_sorted)

    destroy_process_group()


##### #####