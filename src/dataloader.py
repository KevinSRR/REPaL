import os
import json
from tqdm import tqdm
from utils import (
    CKPT2PLM,
    get_so_tagged_example,
    prompt_snowball,
    find_sublist,
    pad_or_truncate_tensor,
)
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import string
from copy import deepcopy

class REDataLoader:
    def __init__(self, args):
        """
        Note the tokens in raw_data are already lower cased if required. 
        Note all the words in self.rel_info, self.rel2def_prompt, self.rel2def_paraphrased_prompts are not lower cased even if required, so further lower case will be needed during tokenization. 
        """
        self.args = args

        # if this is for bert base uncased, still need make sure do_lower_case=True
        self.tokenizer = CKPT2PLM[args.pretrained_lm][0].from_pretrained(args.pretrained_lm)
        
        # predefine the update the following attributes
        self.rel2id, self.id2rel, self.rel_info = {}, {}, {} # -1 not in self.id2rel but -1 will be assigned as relation ids to oov relations (all from distant dataset)
        self.max_length = 0 # records max encoded length (including prompts) across all train, val, test, distant dataset divisions

        self.train_processed, self.val_processed, self.test_processed, self.distant_processed = {'raw_data': None, 'entPairMention2localExIds': None, 'rel2localExIds': None, 'processed_data': None}, {'raw_data': None, 'entPairMention2localExIds': None, 'rel2localExIds': None, 'processed_data': None}, {'raw_data': None, 'entPairMention2localExIds': None, 'rel2localExIds': None, 'processed_data': None}, {'raw_data': None, 'entPairMention2localExIds': None, 'rel2localExIds': None, 'processed_data': None}
        # aggregate all the processed data for general reference
        self.map_processed = {
            'train': self.train_processed,
            'val': self.val_processed,
            'test': self.test_processed,
            'distant': self.distant_processed
        }
        self.read_data(dataset_dir=args.dataset_dir, train_file=args.train_file, val_file=args.val_file, test_file=args.test_file, distant_file=args.distant_file, rel2id_file=args.rel2id_file, id2rel_file=args.id2rel_file, rel_info_file=args.rel_info_file, cache_sub_dir=args.cache_sub_dir)


        # read and organize prompt information for each relation (except for distant relations)
        self.rel2def_prompt, self.rel2def_paraphrased_prompts = {}, {} # from rel to string (relation definiton prompt), from rel to list of strings (paraphrased relation definition prompts) respectively
        self.prepare_def_prompt_and_paraphrased_prompts(overwrite_rel_info_file=False)


    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, val_file, test_file, distant_file, rel2id_file, id2rel_file, rel_info_file, cache_sub_dir):
        """
        Read variables and data, processs data into tensors and store all of them to self attributes. 

        Args: 
            dataset_dir: the dataset directory where train_file, val_file, test_file, distant_file, rel2id_file, id2rel_file, rel_info_file (json files) and cache_sub_dir (dir) lie within.
            cache_sub_dir: stores the cached processed data and tensors. If it does not store them, processing and saving will be conducted.

        Returns:
            None. Only updates self attributes.
        """
        self.rel2id = json.load(open(os.path.join(dataset_dir, rel2id_file)))
        self.id2rel = json.load(open(os.path.join(dataset_dir, id2rel_file)))
        self.rel_info = json.load(open(os.path.join(dataset_dir, rel_info_file)))

        
        tmp_raw_data, tmp_entPairMention2localExIds, tmp_rel2localExIds, tmp_processed_data = self.create_dataset(dataset_dir=dataset_dir, div_file=train_file, cache_sub_dir=cache_sub_dir, div='train')
        self.train_processed = {
            'raw_data': deepcopy(tmp_raw_data),
            'entPairMention2localExIds': deepcopy(tmp_entPairMention2localExIds),
            'rel2localExIds': deepcopy(tmp_rel2localExIds),
            'processed_data': deepcopy(tmp_processed_data),
        }

        tmp_raw_data, tmp_entPairMention2localExIds, tmp_rel2localExIds, tmp_processed_data = self.create_dataset(dataset_dir=dataset_dir, div_file=val_file, cache_sub_dir=cache_sub_dir, div='val')
        self.val_processed = {
            'raw_data': deepcopy(tmp_raw_data),
            'entPairMention2localExIds': deepcopy(tmp_entPairMention2localExIds),
            'rel2localExIds': deepcopy(tmp_rel2localExIds),
            'processed_data': deepcopy(tmp_processed_data),
        }

        tmp_raw_data, tmp_entPairMention2localExIds, tmp_rel2localExIds, tmp_processed_data = self.create_dataset(dataset_dir=dataset_dir, div_file=test_file, cache_sub_dir=cache_sub_dir, div='test')
        self.test_processed = {
            'raw_data': deepcopy(tmp_raw_data),
            'entPairMention2localExIds': deepcopy(tmp_entPairMention2localExIds),
            'rel2localExIds': deepcopy(tmp_rel2localExIds),
            'processed_data': deepcopy(tmp_processed_data),
        }

        tmp_raw_data, tmp_entPairMention2localExIds, tmp_rel2localExIds, tmp_processed_data = self.create_dataset(dataset_dir=dataset_dir, div_file=distant_file, cache_sub_dir=cache_sub_dir, div='distant')
        self.distant_processed = {
            'raw_data': deepcopy(tmp_raw_data),
            'entPairMention2localExIds': deepcopy(tmp_entPairMention2localExIds),
            'rel2localExIds': deepcopy(tmp_rel2localExIds),
            'processed_data': deepcopy(tmp_processed_data),
        } 

        # aggregate all the processed data for general reference
        self.map_processed = {
            'train': self.train_processed,
            'val': self.val_processed,
            'test': self.test_processed,
            'distant': self.distant_processed
        }


    # process data and convert to tensors
    def create_dataset(self, dataset_dir, div_file, cache_sub_dir, div):
        """
        Check if cache_sub_dir contains the processed data. If it has, load and return. Otherwise, process, save to cache_sub_dir and return. 

        Args:
            dataset_dir is the dir that contians div_file and cache_sub_dir. 
            div_file: the raw data file name (contain json extension). It contains data to be processed or load.
            cache_sub_dir:  stores the cached processed data and tensors. If it does not store them, processing and saving will be conducted.
            div: div name of the dataset
        Returns:
            raw_data: list of examples where each example is a dict contain tokens, h, t, div and relation (distant data might also contain r which is same as relation)
            entPairMention2localExIds: dict from string to list of indices. key is the entity mention pair separated by # and value is the indices indexing raw_data that has same entity pair (local indexing)
            rel2localExIds: dict from string to list of indices. key is the relation (not id) and value is the indices indexing raw_data that has the same relation (local indexing)
            processed_data: dict of tensors on cpu. key include 'input_ids', 'attention_mask', 'labels'. Prompt (if required to have) is added during encoding. 
        """

        print(f'===Creating dataset from {div_file.split(".")[0]}===')
        cache_dir = os.path.join(dataset_dir, cache_sub_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, f"{div_file.split('.')[0]}.pt")

        entPairMention2localExIds = {}
        rel2localExIds = {}
        raw_data = []
        processed_data = {
            'input_ids': None,
            'attention_mask': None,
            'labels': None,
        }

        if os.path.exists(cache_file):
            print(f"   Cached processed file exists. Loading processed files from {cache_file}...")
            cached_processed_data = torch.load(cache_file)
            return cached_processed_data['raw_data'], cached_processed_data['entPairMention2localExIds'], cached_processed_data['rel2localExIds'], cached_processed_data['processed_data']
        
        else:
            print(f'   Cached processed file does not exist. Reading raw data from {os.path.join(dataset_dir, div_file)} for processing')
            div_data = json.load(open(os.path.join(dataset_dir, div_file)))
            ex_id = 0
            for rel in div_data:
                rel2localExIds[rel] = []

                for rel_instance in div_data[rel]:
                    tokens_instance = rel_instance['tokens']
                    if self.args.lower_case:
                        tokens_instance = [tok.lower() for tok in rel_instance['tokens']]
                        rel_instance['tokens'] = tokens_instance
                    
                    h_ent = " ".join([tokens_instance[i] for i in rel_instance['h'][2][0]])
                    t_ent = " ".join([tokens_instance[i] for i in rel_instance['t'][2][0]])
                    ht_ent = h_ent + "#" + t_ent
                    if ht_ent not in entPairMention2localExIds:
                        entPairMention2localExIds[ht_ent] = []

                    entPairMention2localExIds[ht_ent].append(ex_id)
                    rel2localExIds[rel].append(ex_id)
                    rel_instance.update({'relation': rel, 'div': div})
                    raw_data.append(rel_instance) # for distant data, key 'r' exists (won't matter)

                    ex_id += 1

            print(f'   Converting texts into tensors...')
            input_ids, attention_mask, labels = self.prompt_encode(raw_data=raw_data, div_file=div_file)
            processed_data['input_ids'] = input_ids
            processed_data['attention_mask'] = attention_mask
            processed_data['labels'] = labels

            # save the processing output to cache file
            print(f'   Saving the processed data into {cache_file}')
            torch.save({
                'raw_data': raw_data,
                'entPairMention2localExIds': entPairMention2localExIds,
                'rel2localExIds': rel2localExIds,
                'processed_data': processed_data,
            }, cache_file)


            return raw_data, entPairMention2localExIds, rel2localExIds, processed_data


    def prompt_encode(self, raw_data, div_file):
        """
        Only handles robert and bert encoding. Rewriting is needed when dealing with other LMs.
        Process the examples including added prompts. Make sure all examples are encoded to the same self.args.max_len (if exceeds, truncate the sentence part). 
        Make sure it's lower cased if required by self.args.lower_case. 
        Update self.max_length if new self.max_length is reached across so far processed div_files.
        Assign relation id as -1 to oov relations that are not in self.rel2id. 


        Args:
            raw_data: list of examples where each example is a dict contain tokens, h, t, and relation. 
            div_file: the raw data file name (contain json extension). Only used for printing some output.
        Returns:
            batched input ids, batched attention masks, label (all in tensors in cpu)
        """
        
        input_str_to_encode = []
        labels = []

        for ex_id, example in enumerate(raw_data):
            relation = example['relation']
            labels.append(self.rel2id.get(relation, -1))

            sent = ' '.join(example['tokens'])

            h_ent = " ".join([example['tokens'][i] for i in example['h'][2][0]])
            t_ent = " ".join([example['tokens'][i] for i in example['t'][2][0]])
            prompt = f' So based on the aforementioned context, the relation between {h_ent} and {t_ent} is: '
            if self.args.lower_case:
                prompt = prompt.lower()
            prompt += f'{h_ent} {self.tokenizer.mask_token} {t_ent}' # assume entity mentions already lowered
            
            tmp_sent = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)) # list of int (missing bos and eos)
            tmp_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))

            if len(tmp_sent) + len(tmp_prompt) + 2 > self.max_length:
                self.max_length = len(tmp_sent) + len(tmp_prompt) + 2

            if len(tmp_sent) + len(tmp_prompt) + 2 > self.args.max_len:
                tmp_sent = tmp_sent[:self.args.max_len - 2 - len(tmp_prompt)]
            tmp = tmp_sent + tmp_prompt
            
            sent_w_prompt = self.tokenizer.decode(tmp, skip_special_tokens=False)
            input_str_to_encode.append(sent_w_prompt)

            if ex_id in [0, 10, 20, 30]:
                print(f'===Example input to be encoded: {sent_w_prompt}===')

        print(f'===Max encoded sequence length so far after processing {div_file}: {self.max_length}===')
        labels = torch.tensor(labels, dtype=torch.long) # convert label into tensor of torch.LongTensor type
        encoded_dict = self.tokenizer(input_str_to_encode, add_special_tokens=True, max_length=self.args.max_len, padding='max_length', return_attention_mask=True, truncation=False, return_tensors='pt')
        
        assert encoded_dict['input_ids'].size(0) == len(raw_data), f'batch size does not match between input data({len(raw_data)}) and output tensors({encoded_dict["input_ids"].size(0)})!'
        assert encoded_dict['input_ids'].size(1) == self.args.max_len, f'the seq length does not match between self.args.max_len({self.args.max_len}) and output tensors({encoded_dict["input_ids"].size(1)})'

        return encoded_dict['input_ids'], encoded_dict['attention_mask'], labels


    def prepare_def_prompt_and_paraphrased_prompts(self, overwrite_rel_info_file=False, allow_zero_length_paraphrased_prompts=True):
        """
        Set up the contents in self.rel2def_prompt and self.rel2def_paraphrased_prompts.

        Args:
            overwrite_rel_info_file: If def_prompt paraphrasing is needed (update self.rel_info), whether or not overwrite orignal self.args.dataset_dir/self.args.rel_info file with the updated self.rel_info.
            allow_zero_length_paraphrased_prompts: if set to True, will not start new paraphrasing if 'prompts' exists in self.rel_info but the value is a zero length list; Otherwise, start paraphrasing to generate new paraphrased prompts
        Returns:
            None
        """
        for rel in self.rel_info:
            self.rel2def_prompt[rel] = self.rel_info[rel]['typed_desc_prompt']
        
        for rel in self.rel_info:
            if 'prompts' in self.rel_info[rel] and (len(self.rel_info[rel]['prompts']) > 0 or allow_zero_length_paraphrased_prompts):
                self.rel2def_paraphrased_prompts[rel] = self.rel_info[rel]['prompts']
            else:
                # paraphrase the relation definition prompt 
                self.rel_info = prompt_snowball(self.rel_info, similarity_threshold=75)
                self.rel2def_paraphrased_prompts[rel] = self.rel_info[rel]['prompts']
                if overwrite_rel_info_file:
                    updated_rel_info_save_path = os.path.join(self.args.dataset_dir, self.args.rel_info_file)
                else:
                    updated_rel_info_save_path = os.path.join(self.args.dataset_dir, self.args.rel_info_file.replace('.json', '_updated.json'))
                with open(updated_rel_info_save_path, 'w') as f:
                    json.dump(self.rel_info, f)


    def sample_and_gather_indices(self, div='test', K=5, backup=50, deterministic=False):
        """
        Operate within the same div data. Operate with indices only, not touching actual data or tensors. So need further retrieval based on the output of this function. 

        First sample backup shots for each relation from the div data and leave all the other data for evaluation. 
        Then iterate through each relation in the div data as the target relation in each iteration, sample K shots for the target relation from the backup data as training data.
        Return each iteration's grouped samples. 

        If want deterministic results for reproducibility, set deterministic=True. 
        
        Args:
            div: dataset to construct the data. Among train|val|test|distant
            K: number of shots for each relation
            backup: number of backup data to be sampled. We require: backup >= K
            deterministic: whether to do deterministic sampling or not
        Returns:
            rel_set: list of relations for div data (in sorted order)
            rel2Kshot_locIds: map from relation to list of local indices. Contain sampled K shot indices for each relation. 
            eval_locIds: list of local indices. Contain the data indices indexing div data for evaluation samples. 
            rel2backup_locIds: map from relation to list of local indices. Contain sampled backup data for each relation. 
        """
        assert K <= backup, f"K({K}) > backup({backup}) which is not allowed!"
        
        processed_div_data = self.map_processed[div]

        rel_set = sorted(list(processed_div_data['rel2localExIds'].keys()))
        rel2backup_locIds = {}
        rel2Kshot_locIds = {}
        eval_locIds = []


        for rel_idx, rel in enumerate(rel_set):
            rel_indices = processed_div_data['rel2localExIds'][rel] # list of local indices
            assert backup < len(rel_indices), f"backup ({backup}) >= len(rel_indices) ({len(rel_indices)}) which is not allowed!"
            
            if deterministic: 
                # random.seed(min(self.args.seed, 1) * (rel_idx + 1))
                rel2backup_locIds[rel] = rel_indices[:backup]
            else:
                # gather the back up local indices for this relation and leave all the remaining for evaluation
                rel2backup_locIds[rel] = random.sample(rel_indices, backup)
            for idx in rel_indices:
                if idx not in rel2backup_locIds[rel]:
                    eval_locIds.append(idx)
            
        for rel_idx, rel in enumerate(rel_set):

            if deterministic: random.seed(min(self.args.seed, 1) * (rel_idx + 1))

            rel2Kshot_locIds[rel] = random.sample(rel2backup_locIds[rel], K)
        
        return rel_set, rel2Kshot_locIds, eval_locIds, rel2backup_locIds


    def make_dataloader(self, div, local_indices, batch_size=64, random_sample=True, shuffle=True, binary=True, target_rel=None):
        """
        Based on the provided div and local_indices, construct and return the dataloader. 
        This function is for returning the dataloader for the binary classification model where the input is: sentence + mention of Entity 0 + <mask> + mention of Entity 1 and the model will predict 0 or 1 based on the hidden states of the mask. The tensors are processed in advance so it's quick when calling.  

        Args:
            div: train|val|test, do not include distant (function rewrite needed otherwise)
            local_indices: list of int
            batch_size: 
            random_sample: use RandomSampler or SequentialSampler
            shuffle: 
            binary: if set to True, returned labels will be binarized with positive label as target_rel and all the other labels will be set as negative label
            target_rel: positive relation for binary setting (if binary=True)
        
        Returns:
            pytorch dataloader
        """

        assert div != 'distant', f"div({div}) can not equal to distant in this function!"
        assert (binary and target_rel is not None) or (not binary and target_rel is None), f"Wrong setting for binary({binary}) and target_rel({target_rel})"

        processed_div_data = self.map_processed[div]
        processed_div_tensors = processed_div_data['processed_data']
        
        label_tensors = processed_div_tensors['labels'][local_indices]
        if binary and target_rel is not None:
            target_rel_id = self.rel2id[target_rel]
            for idx in range(label_tensors.size(0)): label_tensors[idx] = 1 if (label_tensors[idx] == target_rel_id) else 0

        tensor_dataset = TensorDataset(processed_div_tensors['input_ids'][local_indices], processed_div_tensors['attention_mask'][local_indices], label_tensors)

        SAMPLER_CLASS = RandomSampler if random_sample else SequentialSampler
        dataset_sampler = SAMPLER_CLASS(tensor_dataset)
        dataset_dataloader = DataLoader(tensor_dataset, sampler=dataset_sampler, batch_size=batch_size)

        return dataset_dataloader

    
    def get_div_indices_w_same_entpair(self, src_div='test', tgt_div='distant', src_indices=None, ):
        """
        Given the example indices in src_indices for the src_div dataset, return all example indices that share the same entity mention pair from tgt_div dataset. 

        Args:
            src_div: div from train|val|test|distant
            tgt_div: div from train|val|test|distant
            src_indices: list of indices indexing instances from the src_div dataset
        Returns:
            src_ent_pairs: list of entity paris ('head#tail') corresponding each example indexed by src_indices (could have repetitions) [already lower cased if required] [include all src_div ent pairs even if they do not have corresponding examples with same entity pairs in the tgt_div dataset]
            tgt_indices: a sorted list of tgt_div instance indices with same entity mention pairs [no index repetition]
            src_ent_pairs2tgt_indices: dict with key as src_ent_pair (only src_ent_pairs that has correpsonding examples in tgt_div dataset with same ent pair mentions) and value as list of tag_div example indices with same entity mention pair as key
        """

        src_ent_pairs = []
        src_ent_pairs2tgt_indices = {}
        tgt_indices = []

        for src_idx in src_indices:
            rel_instance = self.map_processed[src_div]['raw_data'][src_idx]
            tokens_instance = rel_instance['tokens']
            h_ent = " ".join([tokens_instance[i] for i in rel_instance['h'][2][0]])
            t_ent = " ".join([tokens_instance[i] for i in rel_instance['t'][2][0]])
            ht_ent = h_ent + "#" + t_ent
            src_ent_pairs.append(ht_ent)
            if ht_ent in self.map_processed[tgt_div]['entPairMention2localExIds'] and not ht_ent in src_ent_pairs2tgt_indices:
                src_ent_pairs2tgt_indices[ht_ent] = self.map_processed[tgt_div]['entPairMention2localExIds'][ht_ent]
                tgt_indices.extend(src_ent_pairs2tgt_indices[ht_ent])

        return src_ent_pairs, sorted(list(set(tgt_indices))), src_ent_pairs2tgt_indices
    

    def get_div_raw_data_w_indices(self, div='test', div_indices=None):
        """
        Retrieve a list of examples (dict) from div dataset with each example indexed by the div_indices

        Args: 
            div: from train|val|test|distant
            div_indices: list of indices indexing the div dataset
        Returns:
            list of example dict containing keys as "tokens", "relation", "h" (corresponding value is a length-3 list), "t" (corresponding value is a length-3 list), "div" (source div for the example)
        """

        return [self.map_processed[div]['raw_data'][div_idx] for div_idx in div_indices]


    def print_div_raw_data_w_indices(self, div_list=['test'], local_indices=[0], indent_char='', example_list=None):
        """
        If example_list is None, print each example from div (each elt from div_list) dataset indexed by local_index (each elt from local_indices) 
        If example_list is not None, print the examples in the example_list
        Args:
            div_list: list of div (from train|val|test|distant) for each example to print
            local_indices: list of local index indexing the corresponding div dataset from div_list
            indent_char: indentation for printing
            example_list: list of examples (dict) with keys as tokens, h, t, and relation
        Return:
            Only printing. Returns None.
        """
        
        if example_list is None:
            for div, local_index in zip(div_list, local_indices):
                example = self.map_processed[div]['raw_data'][local_index]
                relation = example['relation']
                print(f"{indent_char}Div: {div}\tLoc Idx: {local_index}\tRelation: {relation}")
                print(f"{indent_char}Tagged Sent: {get_so_tagged_example(example=example, h_type=None, t_type=None)}")
                print('\n')
        else:
            for example_id, example in enumerate(example_list):
                relation = example['relation']
                print(f"{indent_char}Div: Not Given\tRelative Idx: {example_id}\tRelation: {relation}")
                print(f"{indent_char}Tagged Sent: {get_so_tagged_example(example=example, h_type=None, t_type=None)}")
                print('\n')

    
    def make_dataloader_runtime_NLIBased(self, div2local_indices, rel_prompts, batch_size, div2assigned_labels=None, random_sample=False, target_rel=None, ckpt_sub_folder='simi_NLIBased_ckpt/', ckpt_file_suffix='', train_examples=None, train_assigned_labels=None, rank=None, load_or_save_cache=True):
        """
        Retrieve examples from the provided dataset div and indices information. Process them (if ckpt file exists, load and return) and store in ckpt. In processing, concatenate the sentence with each relation prompt (separated by sep tokens) for later NLI inference. Construct binary classification labels if not provided based on target_rel relation.

        Note: all values in the returned dataloader are not moved to any GPU device in this function. 

        [Major Update]: if train_examples and train_assigned_labels are given, it will override the indices parameters and the dataloader will be directly generated by the given train_examples and train_assigned_labels. 

        [Major Update]: if load_or_save_cache is set to False, will not load data from cached files if these files exist. After processing, will not save to cached files if cached files do not exists. If load_or_save_cache is set to True (by default), will load cached files if these files exist. After processing, will save to cached files if cached files do not exists.

        Args:
            div2local_indices: dict from div to list of local indices for div dataset
            rel_prompts: list of strings, each string is with entity placeholders (e.g., "<ENT0>" and "<ENT1>")
            batch_size: batch size of the returned dataloader
            div2assigned_labels: dict from div to list of assigned labels (each label is a one dim tensor) corresponding to div2local_indices 
            random_sample: random sampler for the dataloader or not
            target_rel: if div2assigned_labels is None, we will construct the binary labels to return according to this parameter
            train_examples: list of example dicts with keys as tokens, h, t, relation
            train_assigned_labels: list of assigned labels (each label is a one dim tensor) corresponding to each example in train_examples
        Return: 
            torch dataloader (all tensors):
                input_ids: (#examples, #rel_prompts, max_len)
                attention_mask: (#examples, #rel_prompts, max_len)
                assigned_labels: (#examples, )
        """

        # default ckpt file for processed data of target_rel
        #   if it exists, load and use; otherwise, process and store
        similarity_ckpt_file = os.path.join(self.args.dataset_dir, self.args.cache_sub_dir, ckpt_sub_folder, f'{target_rel}{ckpt_file_suffix}.pt')
        if os.path.exists(similarity_ckpt_file) and load_or_save_cache:
            processed_data = torch.load(similarity_ckpt_file)
            input_ids, attention_mask, assigned_labels = processed_data['input_ids'], processed_data['attention_mask'], processed_data['assigned_labels']

            if rank is not None: 
                relative_ids = processed_data['relative_ids']
                tensor_dataset = TensorDataset(input_ids, attention_mask, assigned_labels, relative_ids)
            else:
                tensor_dataset = TensorDataset(input_ids, attention_mask, assigned_labels)
            SAMPLER_CLASS = RandomSampler if random_sample else SequentialSampler
            if rank is None:
                dataset_sampler = SAMPLER_CLASS(tensor_dataset)
            else:
                dataset_sampler = DistributedSampler(tensor_dataset, shuffle=random_sample)
            dataset_dataloader = DataLoader(tensor_dataset, sampler=dataset_sampler, batch_size=batch_size, pin_memory=True)
            return dataset_dataloader
        
        
        relation_prompts = deepcopy(rel_prompts)
        
        if train_examples is None:
            div_raw_data = []
            assigned_labels = []
            for div, local_indices in div2local_indices.items():
                for local_idx in local_indices:
                    div_raw_data.append(self.map_processed[div]['raw_data'][local_idx])
                
                if div2assigned_labels is not None:
                    assigned_labels.extend(div2assigned_labels[div])
            
            if div2assigned_labels is None:
                assert target_rel is not None, 'div2assigned_labels and target_rel can not be None at the same time'

                for raw_example in div_raw_data:
                    if raw_example['relation'] == target_rel:
                        assigned_labels.append(torch.tensor([1], dtype=torch.long))
                    else:
                        assigned_labels.append(torch.tensor([0], dtype=torch.long))
        else:
            div_raw_data = train_examples
            assert train_assigned_labels is not None, "assigned_labels cannot be None given train_examples is not None!"
            assigned_labels = train_assigned_labels
        
        data_inputs = []

        for i, raw_example in enumerate(tqdm(div_raw_data, desc='Encoding for NLIBased Similarity Measurement')):
            head_ent_mention = " ".join([raw_example['tokens'][tok_id] for tok_id in raw_example['h'][2][0]])
            tail_ent_mention = " ".join([raw_example['tokens'][tok_id] for tok_id in raw_example['t'][2][0]])

            sent = " ".join(raw_example['tokens'])

            example_inputs = []

            for j, rel_prompt in enumerate(relation_prompts):
                if self.args.lower_case:
                    rel_prompt = rel_prompt.lower().replace('<ent0>', '<ENT0>').replace('<ent1>', '<ENT1>')
                
                rel_prompt_filled_w_entities = rel_prompt.replace('<ENT0>', head_ent_mention).replace('<ENT1>', tail_ent_mention)
                rel_prompt_filled_w_entities = f"{sent}{2 * self.tokenizer.sep_token}{rel_prompt_filled_w_entities}"

                inputs = self.tokenizer([rel_prompt_filled_w_entities], add_special_tokens=True, max_length=self.args.max_len, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')

                example_inputs.append(inputs)
            
            data_inputs.append(example_inputs)

        input_ids, attention_mask = None, None
        input_ids = torch.stack(
            [
                torch.stack([
                   j['input_ids'][0] for j in i
                ]) for i in data_inputs
            ]
        ) # (#examples, #rel_prompts, max_len)

        attention_mask = torch.stack(
            [
                torch.stack([
                   j['attention_mask'][0] for j in i
                ]) for i in data_inputs
            ]
        ) # (#examples, #rel_prompts, max_len)

        assigned_labels = torch.cat(assigned_labels, dim=0) # (#examples, )

        if rank is not None: relative_ids = torch.arange(len(div_raw_data)) # (#examples, )

        if rank == 0 or rank is None:
            if not os.path.exists(similarity_ckpt_file) and load_or_save_cache:
                directory = os.path.dirname(similarity_ckpt_file)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                if rank is None:
                    torch.save({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'assigned_labels': assigned_labels,
                    }, similarity_ckpt_file)
                else: # rank is 0, will use ddp training
                    torch.save({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'assigned_labels': assigned_labels,
                        'relative_ids': relative_ids,
                    }, similarity_ckpt_file)     

        if rank is None:
            tensor_dataset = TensorDataset(input_ids, attention_mask, assigned_labels)
        else:
            tensor_dataset = TensorDataset(input_ids, attention_mask, assigned_labels, relative_ids)
        SAMPLER_CLASS = RandomSampler if random_sample else SequentialSampler
        if rank is None:
            dataset_sampler = SAMPLER_CLASS(tensor_dataset)
        else:
            dataset_sampler = DistributedSampler(tensor_dataset, shuffle=random_sample)
        dataset_dataloader = DataLoader(tensor_dataset, sampler=dataset_sampler, batch_size=batch_size, pin_memory=True)

        return dataset_dataloader
    



    def make_negative_batch_runtime_NLIBased(self, batch_pos_labels, rel_prompts, train_locIds, train_divs, train_labels, label2relative_ids):
        """
        Based on the batch data's labels (batch_pos_labels), construct the batch data with same batch size but contains random samples with opposite binary labels corresponding to each entry of the given batch data's labels. 
        The sampling is based on the train_locIds, train_divs, train_labels, label2relative_ids. 
        The returned data is still on CPU.

        Returns:
            neg_batch_tensors, same format as the batch tensor for training as:
                {
                    'input_ids': tensors of (#examples, #rel_prompts, max_len),
                    'attention_mask': tensors of (#examples, #rel_prompts, max_len),
                    'assigned_labels': tensors of (#examples),
                }
            target: one dimensional tensors containing 1/-1 for torch's margin ranking loss (1 if the original label is 1 and the constructed negative label is 0, and -1 otherwise)
        """
        relation_prompts = deepcopy(rel_prompts)
        div_raw_data = []

        neg_batch_tensors = {
            'input_ids': None,
            'attention_mask': None,
            'assigned_labels': [],
        }

        target = []
        for example_pos_label in batch_pos_labels:
            if example_pos_label == 1:
                neg_relative_id = random.choice(label2relative_ids['neg'])
                neg_batch_tensors['assigned_labels'].append(torch.tensor([0], dtype=torch.long))
                target.append(torch.tensor([1], dtype=torch.long))
            else:
                neg_relative_id = random.choice(label2relative_ids['pos'])
                neg_batch_tensors['assigned_labels'].append(torch.tensor([1], dtype=torch.long))
                target.append(torch.tensor([-1], dtype=torch.long))
            
            div_raw_data.append(self.map_processed[train_divs[neg_relative_id]]['raw_data'][train_locIds[neg_relative_id]])
        
        data_inputs = []
        for i, raw_example in enumerate(tqdm(div_raw_data, desc='Encoding negative samples for NLIBased Optimization')):
            head_ent_mention = " ".join([raw_example['tokens'][tok_id] for tok_id in raw_example['h'][2][0]])
            tail_ent_mention = " ".join([raw_example['tokens'][tok_id] for tok_id in raw_example['t'][2][0]])

            sent = " ".join(raw_example['tokens'])

            example_inputs = []

            for j, rel_prompt in enumerate(relation_prompts):
                if self.args.lower_case:
                    rel_prompt = rel_prompt.lower().replace('<ent0>', '<ENT0>').replace('<ent1>', '<ENT1>')

                rel_prompt_filled_w_entities = rel_prompt.replace('<ENT0>', head_ent_mention).replace('<ENT1>', tail_ent_mention)
                rel_prompt_filled_w_entities = f"{sent}{2 * self.tokenizer.sep_token}{rel_prompt_filled_w_entities}"

                inputs = self.tokenizer([rel_prompt_filled_w_entities], add_special_tokens=True, max_length=self.args.max_len, padding='max_length', return_attention_mask=True, truncation=False, return_tensors='pt')

                example_inputs.append(inputs)

            data_inputs.append(example_inputs)


        neg_batch_tensors['input_ids'] = torch.stack(
            [
                torch.stack([
                j['input_ids'][0] for j in i
                ]) for i in data_inputs
            ]
        ) # (#examples, #rel_prompts, max_len)

        neg_batch_tensors['attention_mask'] = torch.stack(
            [
                torch.stack([
                   j['attention_mask'][0] for j in i
                ]) for i in data_inputs
            ]
        ) # (#examples, #rel_prompts, max_len)

        neg_batch_tensors['assigned_labels'] = torch.cat(neg_batch_tensors['assigned_labels'], dim=0)

        target = torch.cat(target, dim=0)
        
        return neg_batch_tensors, target