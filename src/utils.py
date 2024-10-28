from transformers import BertTokenizer, RobertaTokenizer, BertModel, RobertaModel, RobertaForSequenceClassification
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import os
from tqdm import tqdm
from nltk import sent_tokenize
from thefuzz import fuzz
import openai
from torch import nn
from model import (
    NLIBasedSimilarityModel,
)
import matplotlib.pyplot as plt
from collections import defaultdict

# for using the program argument '--pretrained_lm' to index the corresponding model and tokenizer class 
CKPT2PLM = {
    'roberta-large': (RobertaTokenizer, RobertaModel),
    'bert-base-uncased': (BertTokenizer, BertModel),
    'roberta-large-mnli': (RobertaTokenizer, RobertaForSequenceClassification)
}

SIMI_MODEL_CLASS = {
    'NLIBasedSimilarityModel': NLIBasedSimilarityModel,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_binary_f1(preds, labels):
    """
    Compute the binary classification prec, recall, f1 score

    Args:
        preds: a numpy array in the shape of (batch size, ),
        labels: a numpy array in the shape of (batch size, )
    Return:
        prec, recall, f1, support
    """

    prec, recall, f1, support = precision_recall_fscore_support(
        y_true=labels,
        y_pred=preds,
        average='binary'
    )

    return prec, recall, f1, support



def get_so_tagged_example(example, h_type=None, t_type=None, compact=False):
    """
    Given an example in the format of {'tokens': List[Str], 'h': [Any, Any, [inclusive list of token indices], 't': [Any, Any, [inclusive list of token indices]]}, return the sentence with head entity mention and tail entity mention tagged out by '<ENT0>'-'</ENT0>' and <ENT1>'-'</ENT1>' respectively. 
    If h_type and t_type is not None, will enrich the tag with the input type information. 
    If compact==Ture, will have no space between the prefix/suffix entity tags and entity mentions. Otherwise, will have one space.
    Note here ENT0 corresponds to the head entity and ENT1 corresponds to the tail entity.

    Args:
        example: dict
        h_type: Str
        t_type: Str
        compact: Bool
    """

    try:
        token_list = example['tokens']
        subj_span = [example['h'][2][0][0], example['h'][2][0][-1]]
        obj_span = [example['t'][2][0][0], example['t'][2][0][-1]]
        token_list_for_print = []
    except:
        print(example['h'])
        print(example['t'])
        raise Exception
    
    for tok_id, tok in enumerate(token_list):
        if tok_id == subj_span[0]: 
            if h_type is not None and t_type is not None:
                token_list_for_print.append(f"<ENT0_{h_type}>")
            else:
                token_list_for_print.append(f"<ENT0>")

        if tok_id == obj_span[0]: 
            if h_type is not None and t_type is not None:
                token_list_for_print.append(f"<ENT1_{t_type}>")
            else:
                token_list_for_print.append(f"<ENT1>")
        
        if not compact:
            token_list_for_print.append(tok)
        else:
            if tok_id == subj_span[0] or tok_id == obj_span[0]:
                token_list_for_print[-1] = token_list_for_print[-1] + tok
            else:
                token_list_for_print.append(tok)

        if tok_id == subj_span[1]: 
            if not compact:
                if h_type is not None and t_type is not None:
                    token_list_for_print.append(f"</ENT0_{h_type}>")
                else:
                    token_list_for_print.append(f"</ENT0>")
            else:
                if h_type is not None and t_type is not None:
                    token_list_for_print[-1] = token_list_for_print[-1] + f"</ENT0_{h_type}>"
                else:
                    token_list_for_print[-1] = token_list_for_print[-1] + f"</ENT0>"
        if tok_id == obj_span[1]: 
            if not compact:
                if h_type is not None and t_type is not None:
                    token_list_for_print.append(f"</ENT1_{t_type}>")
                else:
                    token_list_for_print.append(f"</ENT1>")
            else:
                if h_type is not None and t_type is not None:
                    token_list_for_print[-1] = token_list_for_print[-1] + f"</ENT1_{t_type}>"
                else:
                    token_list_for_print[-1] = token_list_for_print[-1] + f"</ENT1>"

    return ' '.join(token_list_for_print)



def find_sublist(a, b):
    """
    Return the leftmost starting index to a that b serves as a sublist 

    Args:
        a: a list or array
        b: a list or array with same type as a
    Returns:
        the leftmost starting index indexing a that b serves as a sublist (within a)
            or return None when b is not a sublist of a
    """
    for l in range(len(a)):
        if a[l: l+len(b)] == b:
            return l

    return None


def pad_or_truncate_tensor(input_tensor, length, pad_value=-100):
    """
    If the dimension size of input_tensor is > length, truncate. If the dimension size is < length, pad with pad_value at the end of input_tensor.

    Args:
        input_tensor: should be a one-dimensional tensor
        length: 
    Returns:
        the padded/truncated/unchanged input_tensor with length as the dimension size
    """
    current_length = input_tensor.size(0)
    if current_length < length:
        # pad the tensor to the desired length
        padding = torch.full((length - current_length, ) + input_tensor.shape[1:], pad_value, dtype=input_tensor.dtype)
        padded_tensor = torch.cat((input_tensor, padding), dim=0)
        return padded_tensor
    elif current_length > length:
        # truncate the tensor to the desired length
        truncated_tensor = input_tensor[:length]
        return truncated_tensor
    else:
        # the tensor already has the desired length
        return input_tensor
    




### code for prompt paraphrasing ###
TRANSFORMATIONS_SENT = [['', ''], ['a ', ''], ['the ', '']]
TRANSFORMATIONS_ENT = [['', ''], ['being', 'is'], ['being', 'are'], ['ing', ''], ['ing', 'e']]
class GPT3:
    def __init__(self):
        openai.api_key = ''

    def call(self,
             prompt,
             engine="gpt-3.5-turbo-instruct",
             temperature=1.,
             max_tokens=50,
             top_p=1.,
             frequency_penalty=0,
             presence_penalty=0,
             logprobs=0,
             n=1):
        return openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            n=n)
    
def get_n_ents(prompt):
    """
    Return number of entities in this prompt
    """
    n = 0
    while f'<ENT{n}>' in prompt:
        n += 1
    return n

def get_sent(prompt, ent_tuple):
    """
    Replace the entity placeholders in the prompt with the ent mentions from ent_tuple
    """
    sent = prompt
    for idx, ent in enumerate(ent_tuple):
        sent = sent.replace(f'<ENT{idx}>', ent)

    return sent

def fix_prompt_style(prompt):
    """
    Make sure the prompt string ends with ' .' and if its first char is alphabetical, make it upper cased

    Return:
        formated prompt string
    """
    prompt = prompt.strip('.').strip()
    if prompt[0].isalpha():
        prompt = prompt[0].upper() + prompt[1:]

    return prompt + ' .'

def get_paraphrase_prompt(llm, prompt, ent_tuple):
    assert get_n_ents(prompt) == len(ent_tuple), f"get_n_ents(prompt)({get_n_ents(prompt)}) != len(ent_tuple)({len(ent_tuple)}), where prompt={prompt}, ent_tuple={ent_tuple}"

    # ent_tuple = [ent.lower() for ent in ent_tuple]
    sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)

    for _ in range(5):
        raw_response = llm.call(prompt=f'paraphrase:\n{sent}\n')

        para_sent = raw_response['choices'][0]['text']
        # para_sent = sent_tokenize(para_sent)[0] # take the first sentence 
        # para_sent = para_sent.strip().strip('.').lower()
        para_sent = para_sent.strip().strip('.')
        
        print('para_sent:', para_sent)

        prompt = para_sent
        valid = True
        for idx, ent in enumerate(ent_tuple):
            # for trans_sent in TRANSFORMATIONS_SENT:
            #     for trans_ent in TRANSFORMATIONS_ENT:
            #         if prompt.count(f'<ENT{idx}>') == 0:
            #             transed_prompt = prompt.replace(*trans_sent)
            #             transed_ent = ent.replace(*trans_ent)
            #             if transed_prompt.count(transed_ent) == 1:
            #                 prompt = transed_prompt.replace(transed_ent, f'<ENT{idx}>')
            if prompt.count(f'<ENT{idx}>') == 0:
                prompt = prompt.replace(ent, f'<ENT{idx}>')

            if prompt.count(f'<ENT{idx}>') != 1:
                valid = False
                break

        if valid:
            return prompt
    return None

def search_prompts(init_prompts, seed_ent_tuples, similarity_threshold):
    """
    stop when in a new iteration, no new prompt candidate is generated or no new generated prompt candidate is added to the prompts or the prompts has a length >= 10
    Note newly generated prompt that has the max similarity (over all prompts) >= similarity_threshold will not be added to prompts
    """
    llm = GPT3()

    cache = {}
    prompts = []

    while True:
        new_prompts = []
        for prompt in init_prompts + init_prompts + prompts:
            for ent_tuple in seed_ent_tuples:
                # ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                request_str = f'{prompt} ||| {ent_tuple}'
                if request_str not in cache or prompt in init_prompts:
                    cache[request_str] = get_paraphrase_prompt(
                        llm=llm,
                        prompt=prompt,
                        ent_tuple=ent_tuple,
                    )

                para_prompt = cache[request_str]
                print(f'prompt: {prompt}\tent_tuple: {ent_tuple}'
                      f'\t-> para_prompt: {para_prompt}')
                if para_prompt is not None and \
                        para_prompt not in init_prompts + prompts:
                    new_prompts.append(para_prompt)

            if len(set(prompts + new_prompts)) >= 20:
                break
            
        if len(new_prompts) == 0:
            break
        else:
            # prompts.extend(new_prompts)
            flag = False # indicate whether there is a prompt in new_prompts added to prompts
            for new_prompt in sorted(new_prompts, key=lambda t: len(t)):
                if len(prompts) != 0:
                    # max_sim = max([fuzz.ratio(new_prompt, prompt) for prompt in prompts])
                    max_sim = max([fuzz.ratio(new_prompt.lower(), prompt.lower()) for prompt in prompts])
                    print(f'-- {new_prompt}: {max_sim}')
                # if len(prompts) == 0 or \
                #         max([fuzz.ratio(new_prompt, prompt) for prompt in prompts]) < similarity_threshold:
                if len(prompts) == 0 or \
                        max([fuzz.ratio(new_prompt.lower(), prompt.lower()) for prompt in prompts]) < similarity_threshold:
                    prompts.append(new_prompt)
                    flag = True

            
            prompts = list(set(prompts))
            prompts.sort(key=lambda s: len(s))

            if len(prompts) >= 10 or flag == False:
                break

    # for i in range(len(prompts)):
    #     prompts[i] = fix_prompt_style(prompts[i])
    
    return prompts

def fix_prompts(prompts, ent_tag_list=['<ENT0>', '<ENT1>']):
    """
    Insert a space if there is any char that directly pre-fix any entity tag (aka. placeholder)
    """
    def fix_prompt(prompt, ent_tag_list=['<ENT0>', '<ENT1>']):
        for ent_idx, ent_tag in enumerate(ent_tag_list):
            start_idx = prompt.find(ent_tag)
            if start_idx == 0 or prompt[start_idx - 1] == ' ':
                continue
            else:
                prompt = prompt[:start_idx] + ' ' + prompt[start_idx:]
        return prompt
    
    if isinstance(prompts, list):
        return [fix_prompt(prompt, ent_tag_list=ent_tag_list) for prompt in prompts]
    else:
        return fix_prompt(prompt=prompts, ent_tag_list=ent_tag_list)
    

def prompt_snowball(rel_info, similarity_threshold=75):
    """
    Paraphrase the relation definition prompts
    """
    print('\n' + '=' * 20 + '\n')
    print('original rel_info: ')
    print(rel_info)

    for rel, info in rel_info.items():
        if info['typed_desc_prompt'] is None:
            continue

        # info['typed_desc_prompt'] = fix_prompt_style(info['typed_desc_prompt'])

        if 'prompts' not in info or len(info['prompts']) == 0:
            info['prompts'] = search_prompts(
                init_prompts=[info['typed_desc_prompt']],
                seed_ent_tuples=[['<ENT0>', '<ENT1>']],
                similarity_threshold=similarity_threshold,
            )

            info['prompts'] = fix_prompts(prompts=info['prompts'], ent_tag_list=['<ENT0>', '<ENT1>'])
    
    print('\n\n')
    print('updated rel_info: ')
    print(rel_info)
    print('\n' + '=' * 20 + '\n') 
    return rel_info


def update_model_state_dict(model, target_state_dict):
    """
    Update the model (could be DataParallel wrapped or not)'s state_dict with target_state_dict (could have prefix ``module.'' or not).
    It's in-place update.
    """

    model_state_dict = model.state_dict()
    
    # adjust the keys in the loaded state dictionary to match the model's state dict keys
    new_state_dict = {}
    for k, v in target_state_dict.items():
        if isinstance(model, nn.DataParallel):
            name = k if k.startswith('module.') else 'module.' + k # add 'module.' prefix if it's missing and the model is wrapped in DataParallel
        else:
            name = k[7:] if k.startswith('module.') else k # remove 'module.' prefix if it's present and the model is not wrapped in DataParallel
        new_state_dict[name] = v

    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict, strict=True)





def average_epoch_values(x, y):
    # check if x and y are of the same length
    if len(x) != len(y):
        raise ValueError("Lists x and y must have the same length.")

    # create a dictionary to group y values by x values
    groups = defaultdict(list)
    for epoch_id, value in zip(x, y):
        groups[epoch_id].append(value)

    # calculate averages for each group
    averages = {epoch_id: sum(values) / len(values) for epoch_id, values in groups.items()}

    # prepare the final lists
    sorted_unique_x = sorted(list(averages.keys()))
    averaged_y = [averages[x_elt] for x_elt in sorted_unique_x]

    return sorted_unique_x, averaged_y


import kneed
from kneed import KneeLocator
def get_elbow_x_value(y_values, x_values=None, y_threshold=0.2, S=1, curve="convex", direction="decreasing", online=False):
    """
    Given a curve with x_values and y_values, find the elbow/knee x coordinate of the curve.
    We consider the curve starting from the point where the y_value first <= y_threshold and ignore the part of curve before this point.

    Note: One way to improve this is to first find the max of the first 3 epochs and start from there. (this might be robust to the cases where at the start of training, the loss goes up in the middle and then goes down)

    
    Args: 
        x_values: list of x coordinates of curve
        y_values: list of y coordinates of curve
        y_threshold: threshold for which each y_value is compared with until first find a y_value <= y_threshold
        S: parameter of calling kneed.KneeLocator
        curve: parameter of calling kneed.KneeLocator, whether the curve is convex or concave
        direction: parameter of calling kneed.KneeLocator, whether the slope of curve is positive or negative
        online: parameter of calling kneed.KneeLocator

    Returns: 
        elbow_x_value: the x coordinate of the elbow/knee point
        kneedle: instance of KneeLocator (contain the results x, y value and can also called to visualize)
    """

    if x_values is None: x_values = list(range(len(y_values)))
    assert len(x_values) == len(y_values), "x and y series not matched!"

    # average values that are from the same epoch
    x_values, y_values = average_epoch_values(x_values, y_values)
    print(f"Average values from the same epochs which results in x_values={x_values}, y_values={y_values}")

    elbow_x_value = x_values[-1] # default to last value
    start_idx = 0
    for idx, y_value in enumerate(y_values):
        start_idx = idx + 0
        if y_value <= y_threshold: break
    
    if start_idx == len(y_values) - 1: 
        print(f"The y values do not go under {y_threshold}, so taking the last x value (x={elbow_x_value}) as the elbow point")
        return elbow_x_value, None

    trimmed_x_values, trimmed_y_values = x_values[start_idx:], y_values[start_idx:]
    kneedle = KneeLocator(trimmed_x_values, trimmed_y_values, S=S, curve=curve, direction=direction, online=online)

    if kneedle.elbow is None:
        print(f"Kneed didn't find the elbow. So taking the last x value (x={elbow_x_value}) as the elbow point")
        return elbow_x_value, kneedle
    else:
        elbow_x_value = kneedle.elbow
        print(f"Founded elbow at x={elbow_x_value}")
        
    return elbow_x_value, kneedle


from sentence_transformers import SentenceTransformer
def get_SBERT_embedding(sent_list, sbert_model='roberta-large-nli-stsb-mean-tokens', batch_size=None, normalize_embeddings=True):
    """
    Encode the given sentences to get their sentence embeddings with sentence transformer.

    Args:
        sent_list: list of string with each string as a sentence to get the sentence embedding
        sbert_model: ckpt name or path for the sentence transformer
    Returns:
        numpy array of shape (|sent_list|, hidden size of sentence embedding)

    Note: remaining bug, after calling this function, part of the GPU memory is still not freed
    """

    model = SentenceTransformer(sbert_model)
    model = model.cuda()

    if batch_size is None:
        embeddings = []
        for sent_id, sent in tqdm(enumerate(sent_list), desc='Encoding sentences with SentenceTransformer'):
            emb = model.encode(sent, normalize_embeddings=normalize_embeddings)
            embeddings.append(emb)
        embeddings = np.stack(embeddings, axis=0)
    else:
        embeddings = model.encode(sentences=sent_list, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize_embeddings)
    print('\tShape: ', embeddings.shape)


    # moving sentence transformer back to CPU
    model = model.cpu()

    torch.cuda.empty_cache()
    return embeddings







###### visualize the predicted prob distribution ###### 

def plot_prediction_distributions(test_labels, predictions, bins=50, save_path='distribution_plot.png'):
    """
    Plots and saves the distribution of predicted probabilities for two classes.

    Args:
        test_labels: Numpy array containing the ground truth labels (0s and 1s) of the test set.
        predictions: Numpy array of shape (#test set, 2) containing the prediction distribution for label 0 and label 1.
        bins: Number of bins to use for the probability ranges.
        save_path: Path to save the plot image.
    """

    # separate the predictions based on ground truth labels
    predictions_label_0 = predictions[test_labels == 0, 1]  # probabilities for true label 0
    predictions_label_1 = predictions[test_labels == 1, 1]  # probabilities for true label 1

    # define bins for probability
    bins = np.linspace(0, 1, bins + 1)

    # count samples in each bin
    hist_label_0, _ = np.histogram(predictions_label_0, bins=bins, density=False)
    hist_label_1, _ = np.histogram(predictions_label_1, bins=bins, density=False)

    # midpoints of bins for plotting
    bin_midpoints = (bins[:-1] + bins[1:]) / 2


    # plotting the curves in separate subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # plot for true label 0
    axs[0].plot(bin_midpoints, hist_label_0, label='True Label 0', color='red', marker='o')
    axs[0].set_ylabel('Number of Samples')
    axs[0].set_title('Predicted Probability Distribution for True Label 0')
    axs[0].grid(True)

    # plot for true label 1
    axs[1].plot(bin_midpoints, hist_label_1, label='True Label 1', color='blue', marker='o')
    axs[1].set_xlabel('Predicted Probability for Being Label 1')
    axs[1].set_ylabel('Number of Samples')
    axs[1].set_title('Predicted Probability Distribution for True Label 1')
    axs[1].grid(True)

    # save the figure
    plt.savefig(save_path, format='png', dpi=300)

    # close the plot
    plt.close()

    return f"Plot saved to {save_path}"





import socket

def find_free_port(start_port=12345, end_port=12400):
    for port in range(start_port, end_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            return port
        except socket.error as e:
            if e.errno == 98:  # port is already in use
                continue
            else:
                raise e  # reraise any other socket errors
    raise RuntimeError("No free port found in the specified range")



def remove_repeats_based_on_inference_ids(data_dict):
    inference_ids = data_dict.get('inference_relative_ids', None)
    first_dim_size = inference_ids.shape[0]
    
    if inference_ids is None or not isinstance(inference_ids, np.ndarray):
        raise ValueError("field 'inference_relative_ids' must be present and be a numpy array.")
    
    _, unique_indices = np.unique(inference_ids, return_index=True)
    unique_indices = np.sort(unique_indices)  # Sort to maintain order

    data_dict['inference_relative_ids'] = inference_ids[unique_indices]


    for key, value in data_dict.items():
        if key == 'inference_relative_ids': continue
        
        if isinstance(value, np.ndarray) and value.shape[0] == first_dim_size:
            data_dict[key] = value[unique_indices]

    return data_dict