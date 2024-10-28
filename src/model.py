from typing import Any
from torch import nn
import torch
from transformers import (
    RobertaForSequenceClassification,
)





class NLIBasedSimilarityModel(nn.Module):
    def __init__(self, args, tokenizer, target_relation, rel_prompts, rel_prompt_weights=None, ):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.target_relation = target_relation
        self.rel_prompts = rel_prompts

        if rel_prompt_weights is None:
            self.rel_prompt_weights = nn.Parameter(torch.ones(len(rel_prompts), dtype=torch.float))
            self.rel_prompt_weights.data /= self.rel_prompt_weights.data.sum()
        else:
            pass
        
        self.bert = RobertaForSequenceClassification.from_pretrained(
            args.pretrained_lm,
            return_dict=True,
            cache_dir=args.huggingface_cache_dir if args.huggingface_cache_dir else None
        )

        # print(self.bert)
        
    def normalize_weights(self, weights):
        normalized_weights = torch.nn.functional.softmax(weights, dim=0)
        return normalized_weights
    
    def forward(self, input_ids, attention_mask, assigned_labels):
        """
        Args:
            input_ids: (batch size, len(rel_prompts), max_len)
            attention_mask: (batch size, len(rel_prompts), max_len)
            assigned_labels: 

        Returns:
            weighted_normalized_nli_logits_by_prompt: (batch size, len(rel_prompts), 3) normalized entailment scores by prompts
            weighted_normalized_nli_logits: (batch size, 3) normalized entailment scores averaged over prompts
            nli_scores: (batch size, 2) normalized scores for binary classification. (batch size, 0) is the neg score,  (batch size, 1) is the positive score
        """

        batch_size = input_ids.size(0)
        len_rel_prompts = input_ids.size(1)
        max_len = input_ids.size(2)

        assert len_rel_prompts == len(self.rel_prompts), f'relation prompt number does not match! In loaded data({len_rel_prompts}) != in model class({len(self.rel_prompts)})'
        
        batch_size_transformed = batch_size * len_rel_prompts
        input_ids_transformed = input_ids.view(batch_size_transformed, -1)
        attention_mask_transformed = attention_mask.view(batch_size_transformed, -1)

        logits_transformed = self.bert(input_ids=input_ids_transformed, attention_mask=attention_mask_transformed).logits # (batch_size_transformed, 3)
        logits = logits_transformed.view(batch_size, len_rel_prompts, -1) # (batch size, len(rel_prompts), 3)



        normalized_nli_logits_by_prompt = nn.functional.softmax(logits, dim=-1) # normalize the prediction logits, (batch size, len(rel_prompts), 3)


        weighted_normalized_nli_logits_by_prompt = normalized_nli_logits_by_prompt
        weighted_normalized_nli_logits = weighted_normalized_nli_logits_by_prompt.mean(dim=1) # aggregate inference of prompts, (batch size, 3)

        nli_neg_scores = torch.sum(weighted_normalized_nli_logits[:, :2], dim=1)
        nli_pos_scores = weighted_normalized_nli_logits[:, 2]
        nli_scores = torch.stack([nli_neg_scores, nli_pos_scores], dim=1) # (batch size, 2)
        return weighted_normalized_nli_logits_by_prompt, weighted_normalized_nli_logits, nli_scores
    




import asyncio
import json
import time
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata

from typing import Any, Optional, Union
import openai
from openai import OpenAI
from openai import AsyncOpenAI
import tiktoken # requires Python >= 3.8

# adjust accordingly based on your needs
DEFAULT_RATE_LIMITS = {
    "gpt-3.5": {
        "max_requests_per_minute": 3500,
        "max_tokens_per_minute": 40000,
    },
    "gpt-4": {
        "max_requests_per_minute": 5000, 
        "max_tokens_per_minute": 600000,
    },
}

def load_jsonl(filename: str):
    """Load a JSONL file into a list of dictionaries."""
    with open(filename, "r") as f:
        erg = [json.loads(item) for item in f.readlines()]
    return erg


def append_to_jsonl(data, filename: str):
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0 # number of tasks started
    num_tasks_in_progress: int = 0  # number of tasks in progress, script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = (
        0  # used to cool off after hitting rate limits
    )
    num_prompt_tokens_used: int = 0
    num_completion_tokens_used: int = 0

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: Union[int, str]
    request_json: dict # {'id': int/str, 'prompt': str, 'messages': list[dict], }
    attempts_left: int
    metadata: dict # dict containing all the hyperparameters
    result: list = field(default_factory=list)

    def __post_init__(self):
        # load metadata from metadata object for more readable code
        self.model = self.metadata["GPT_model"]
        self.temperature = self.metadata["temperature"]
        self.max_tokens = self.metadata["max_tokens"]
        self.top_p = self.metadata["top_p"]
        self.frequency_penalty = self.metadata["frequency_penalty"]
        self.presence_penalty = self.metadata["presence_penalty"]
        self.seed = self.metadata["seed"]
        self.logprobs = self.metadata["logprobs"]
        self.n = self.metadata["n"]
        self.token_encoding_name = self.metadata["token_encoding_name"]
        self.api_key = self.metadata['api_key']
        
        self.metadata["task_id"] = self.task_id
        if 'prompt' in self.request_json:
            self.metadata["prompt"] = self.request_json["prompt"]
        else:
            self.metadata["messages"] = self.request_json["messages"]

        # self.client = OpenAI(api_key=self.api_key)
        self.client = AsyncOpenAI(api_key=self.api_key)

    def num_tokens_consumed_from_request(
        self,
    ):
        """Count the number of tokens in the request. Only supports completion and chat completion requests."""
        encoding = tiktoken.get_encoding(self.token_encoding_name)
        # if completions request, tokens = prompt + n * max_tokens
        if 'prompt' in self.request_json or 'messages' in self.request_json: # if prompt in request, assume we call completion instead of chat completion
            completion_tokens = self.n * self.max_tokens

            if 'prompt' in self.request_json: # normal completion model
                prompt = self.request_json["prompt"]
                if isinstance(prompt, str):  # single prompt
                    prompt_tokens = len(encoding.encode(prompt))
                    num_tokens = prompt_tokens + completion_tokens
                    return num_tokens
                elif isinstance(prompt, list):  # multiple prompts
                    prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                    num_tokens = prompt_tokens + completion_tokens * len(prompt)
                    return num_tokens
                else:
                    raise TypeError(
                        'Expecting either string or list of strings for "prompt" field in completion request'
                    )
            else: # chat completion
                num_tokens = 0
                for message in self.request_json['messages']:
                    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                    for key, value in message.items():
                            num_tokens += len(encoding.encode(value))
                            if key == "name":  # if there's a name, the role is omitted
                                num_tokens -= 1  # role is always required and always 1 token
                num_tokens += 2  # every reply is primed with <im_start>assistant
                return num_tokens + completion_tokens


    async def call_api(
        self,
        retry_queue: asyncio.Queue,
        save_filepath: Optional[str],
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""

        error = None
        error_filepath = (
            ".".join(save_filepath.split(".")[:-1]) + "_errors.jsonl"
        )


        params = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=self.n,
        )
        if 'prompt' in self.request_json: 
            params['prompt'] = self.request_json["prompt"]
            params['logporbs'] = self.logprobs
        else:
            params['seed'] = self.seed
            params['messages'] = self.request_json['messages']


        try:
            if 'prompt' in self.request_json:
                response = await self.client.completions.create(**params)
            else:
                response = await self.client.chat.completions.create(**params)
            
            status_tracker.num_prompt_tokens_used += response.usage.prompt_tokens
            status_tracker.num_completion_tokens_used += (
                response.usage.completion_tokens
            )
        except Exception as e:  # handling request errors
            status_tracker.num_other_errors += 1
            error = e
            response = None

            if e == openai.RateLimitError:
                status_tracker.time_of_last_rate_limit_error = time.time()
                status_tracker.num_rate_limit_errors += 1


        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                print(
                    f"Request {self.task_id} failed after all attempts. Saving errors."
                )
                data = (
                    [[str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [[str(e) for e in self.result]]
                )
                append_to_jsonl(data, error_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
            return None
        else:
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

            if save_filepath:
                tmp_raw_request = response.model_dump()
                tmp_raw_request["request"] = params
                tmp_raw_request["task_id"] = self.task_id
                append_to_jsonl(tmp_raw_request, save_filepath)
            return response



class GPTCompletionModel:
    def __init__(self, GPT_model='gpt-4', save_filepath=None, api_key=None, max_attempts=5):
        """ 
        Need to manually modify the hyperparameters for calling the openai api.

        Args:
            GPT_model: openai model ckpt
            save_filepath: save path of the output results which includes input information for api calls (error file will be derived from this file) and input which contains unfinished tasks information if self.check_finished_ids is True
                If this arg is set to None, will not save output results and will not read unfinished tasks information.
                Note that unfinished tasks and input_list should all contain a key in the dict which is 'task_id'.
            api_key: openai api key
            max_attempts: 
        """
        self.GPT_model = GPT_model
        self.save_filepath = save_filepath
        self.api_key = api_key # openai api token
        openai.api_key = api_key


        # derive rate limits based on GPT_model and DEFAULT_RATE_LIMITS
        self.max_requests_per_minute, self.max_tokens_per_minute = None, None
        for model_name in DEFAULT_RATE_LIMITS:
            if GPT_model.startswith(model_name):
                self.max_requests_per_minute = float(DEFAULT_RATE_LIMITS[model_name]["max_requests_per_minute"])
                self.max_tokens_per_minute = float(DEFAULT_RATE_LIMITS[model_name]["max_tokens_per_minute"])
        if self.max_requests_per_minute is None or self.max_tokens_per_minute is None: 
            print('Using default RPM and TPM as model not matched in DEFAULT_RATE_LIMITS...')
            self.max_requests_per_minute, self.max_tokens_per_minute = 500, 10000

        self.max_attempts = int(max_attempts)


        # some other hyperparams for calls
        self.check_finished_ids = False
        self.finished_ids = None
        self.seed = None
        self.temperature = 1.
        self.max_tokens = 50
        self.top_p = 1
        self.frequency_penalty = 0
        # self.presence_penalty = 0.15
        self.presence_penalty = 0
        self.logprobs=0
        self.n = 1
        self.token_encoding_name = "cl100k_base" # For GPT3.5 and GPT4, see https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

    def async_calls(self, input_list):
        """
        Conduct asychronous openai api calls on list of inputs in input_list.

        Args:
            input_list: list of input in the form of dict as
                    {
                        'task_id': int/str,
                        'prompt': str, 
                        'messages': list[dict],
                    }
            when you want to use messages as input, do not include the 'prompt' key-value
            This function will only use openaiclient.completions for inputs as in 'prompt' and openaiclient.chat.completions for conversational inputs in 'messages'
            Note that task_id is mandatory for calling api and reading unfinished tasks information. 
        Returns:
            list of openai api call results corresponding the input_list
        """
        return asyncio.run(
            self.async_calls_helper(input_list=input_list)
        )

    async def async_calls_helper(self, input_list):
        """
        Processing a list of API requests in parallel, ensuring that they stay under the rate limits. 
        """
        id_field_getter = lambda x: x['task_id']


        # constants
        seconds_to_pause_after_rate_limit_error = 1
        seconds_to_sleep_after_each_loop = (
            0.001 # 1 ms limits max throughput to 1,000 requests per second
        )

        # initialize trackers
        results_future_list = []
        queue_of_requests_to_retry = asyncio.Queue()
        task_id_generator = (
            self.task_id_generator_function()
        ) # generates integer IDs of 1, 2, 3, ...

        status_tracker = (
            StatusTracker()
        ) # single instance to track a collection of variables

        next_request = None # variable to hold the next request to call

        # initialize available capacity counts
        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        if self.check_finished_ids:
            if self.save_filepath:
                try:
                    with open(self.save_filepath) as file:
                        finished = file.__iter__()
                        finished_tasks = set(
                            [self.get_id_from_finished(json.loads(line)) for line in finished]
                        )
                except FileNotFoundError:
                    finished_tasks = set()
            else:
                assert (
                    self.finished_ids is not None
                ), "If self.check_finished_ids is True and no self.save_filepath is provided for loading finished ids, self.finished_ids must be provided"
                finished_tasks = self.finished_ids
        else:
            finished_tasks = set()

        # initialize flags
        file_not_finished = True # after file is empty, we'll skip reading it


        task_generator = iter(input_list)

        # entering main loop
        while True:
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                elif file_not_finished:
                    try:
                        # get new request
                        next_task = next(task_generator)
                        if id_field_getter is None:
                            next_task_id = next(task_id_generator)
                        else:
                            next_task_id = id_field_getter(next_task)
                        if next_task_id in finished_tasks:
                            # skipping request of next_task_id because it is already finished
                            continue


                        meta_data = {
                            "GPT_model": self.GPT_model,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "top_p": self.top_p,
                            "frequency_penalty": self.frequency_penalty,
                            "presence_penalty": self.presence_penalty,
                            "seed": self.seed,
                            "logprobs": self.logprobs,
                            "n": self.n,
                            "token_encoding_name": self.token_encoding_name,
                            'api_key': self.api_key,
                        }
                        next_request = APIRequest(
                            task_id=next_task_id,
                            request_json=next_task,
                            attempts_left=self.max_attempts,
                            metadata=meta_data,
                        )

                        status_tracker.num_tasks_started += 1

                        status_tracker.num_tasks_in_progress += 1
                    except StopIteration:
                        # request list exhausted
                        file_not_finished = False
                
            
            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity
                + self.max_requests_per_minute * seconds_since_update / 60.0,
                self.max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity
                + self.max_tokens_per_minute * seconds_since_update / 60.0,
                self.max_tokens_per_minute
            )
            last_update_time = current_time


            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.num_tokens_consumed_from_request()
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    task = asyncio.create_task(
                        next_request.call_api(
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=self.save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    results_future_list.append(task)
                    next_request = None # reset next_request to empty
            
            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress <= 0:
                break
            

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_after_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (
                time.time() - status_tracker.time_of_last_rate_limit_error
            )
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (
                    seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                )
                await asyncio.sleep(remaining_seconds_to_pause)

        # after finishing, log final status
        if status_tracker.num_tasks_failed > 0:
            print(
                f'{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {".".join(self.save_filepath.split(".")[:-1]) + "_errors.jsonl"}.'
            )

        if status_tracker.num_rate_limit_errors > 0:
            print(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )

        results = await asyncio.gather(*results_future_list, return_exceptions=True)
        return results

    def task_id_generator_function(self, ):
        """Generate integers 0, 1, 2, and so on."""
        task_id = 0
        while True:
            yield task_id
            task_id += 1

    def get_id_from_finished(self, result: dict):
        return result["task_id"]
    

    def update_call_attributes(self, **kwargs):
        """ 
        Intended to be used to modify openai api calling hyperparameters
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    # chat style call
    def __call__(
        self,
        input, # {'task_id': int, 'prompt': str, 'messages': list[dict]} or str
        prompt_completion_as_chat_completion=False, # whether to conduct prompt completion using the chat completion api call
        response_format=None,
    ):
        """ 
        Conduct one time openai api calling. 

        Args:
            input: dict as {'task_id': int, 'prompt': str, 'messages': list[dict]} or str. If want to call on 'messages' conversations, key 'prompt' should not be included. 
            prompt_completion_as_chat_completion: whether to conduct prompt completion using the chat completion api call
        """
        client = OpenAI(api_key=self.api_key)
        if isinstance(input, str) or 'prompt' in input:
            one_time_content = input if isinstance(input, str) else input['prompt']
            if prompt_completion_as_chat_completion:
                return client.chat.completions.create(
                    model=self.GPT_model,
                    messages=[
                        {
                            "role": "user",
                            "content": one_time_content,
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    seed=self.seed,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                )
            else:
                return client.completions.create(
                    model=self.GPT_model,
                    prompt=one_time_content,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    logprobs=self.logprobs,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                )

        elif 'messages' in input:
            if response_format is not None:
                return client.chat.completions.create(
                    model=self.GPT_model,
                    messages=input['messages'],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    seed=self.seed, # sets to None normally since it's still a beta feature
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    response_format=response_format,
                )
            
            return client.chat.completions.create(
                model=self.GPT_model,
                messages=input['messages'],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                seed=self.seed, # sets to None normally since it's still a beta feature
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                
            )
        else:
            print(f"Unrecognized input: {input.keys()}")



if __name__ == '__main__':
    
    model = GPTCompletionModel(GPT_model="gpt-3.5-turbo", api_key='', save_filepath='./temp_gpt_new.jsonl')
    model.update_call_attributes(max_tokens=3)
    input_list = []
    for i in range(100):
        input_list.append({'task_id': i, 'messages': [{'role': 'user', 'content': f'say {i}'}]})

    current_time = time.time()
    output = model.async_calls(input_list=input_list)
    print("Time consumed with parallel: ", time.time() - current_time)
    

    current_time = time.time()
    for input in input_list:
        output = model(input=input)
    print("Time consumed: ", time.time() - current_time)