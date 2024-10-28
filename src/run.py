import argparse
from dataloader import REDataLoader
from trainer import ModelTrainer
import torch
import os
from multiprocessing import cpu_count
os.environ['TIKTOKEN_CACHE_DIR'] = '' # in case the cache dir is occupied and you don't have access


def main():
    parser = argparse.ArgumentParser(
        description='main',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # running initial synthesis-training or followup synthesis-training
    parser.add_argument(
        '--run_type',
        choices=['initial', 'followup'],
        required=True,
        help="running initial synthesis-training or followup synthesis-training. Must be either 'initial' or 'followup'."
    )

    # data related files to read
    parser.add_argument('--dataset_dir', default='data/fewrel_defon_1/', help='dataset directory')
    parser.add_argument('--train_file', default='train.json', help='training data (under dataset directory)')
    parser.add_argument('--val_file', default='val.json', help='validation data (under dataset directory)')
    parser.add_argument('--test_file', default='test.json', help='test data (under dataset directory)')
    parser.add_argument('--distant_file', default='distant.json', help='distantly supervised data (under dataset directory)')
    parser.add_argument('--rel2id_file', default='rel2id.json', help='map from relation to relation index (under dataset directory)')
    parser.add_argument('--id2rel_file', default='id2rel.json', help='map from relation index to relation (under dataset directory)')
    parser.add_argument('--rel_info_file', default='rel_info_updated.json', help='map from relation to related information (including prompts and definitions)')
    parser.add_argument('--cache_sub_dir', default='cache/', help='cache directory under the dataset_dir storing model ckpt and processed data')


    # LM related hyperparameters
    parser.add_argument('--max_len', type=int, default=256, help='length (including prompt) that documents are padded/truncated to when encoded to tensors')
    parser.add_argument('--pretrained_lm', type=str, default='roberta-large', help='loading checkpoint for pretrained language models [roberta-large-mnli | roberta-large]')
    parser.add_argument('--lower_case', action='store_true', help='whether or not to lower the case of all texts')
    parser.add_argument('--huggingface_cache_dir', default='./', help='absolute path for huggingface cache directory')
    parser.add_argument('--llm_model_ckpt', type=str, default='gpt-4o-2024-05-13', help='model ckpt for calling large language models')

    # training hyperparameters
    parser.add_argument('--seed', type=int, default=0, help='random seed for set up')
    parser.add_argument('--dist_port', type=int, default=12345, help='distributed training port id')
    parser.add_argument('--eval_batch_size', type=int, default=40, help='batch size per GPU for evaluation in the input (in programs, will be adjusted to total eval batch size by multipied with #gpus. Meanwhile, args.train_batch_size_per_device will hold the input.)')
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size per GPU for training in the input (in programs, will be adjusted to total train batch size by multipied with #gpus. Meanwhile, args.eval_batch_size_per_device will hold the input.)')
    parser.add_argument('--unlabel_infer_batch_size', type=int, default=40, help='batch size per GPU for evaluation in the input (in programs, will be adjusted to total eval batch size by multipied with #gpus. Meanwhile, args.unlabel_infer_batch_size_per_device will hold the input.)')
    parser.add_argument('--max_steps', type=int, default=0, help='number of maximal steps for training. If its value > 0, it will override --num_train_epochs')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='number of epochs for training')
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps during training')
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="initial learning rate for Adam")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay if we apply some")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", default=0, type=int, help="linear warmup over warmup_steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm")
    parser.add_argument('--logging_steps', type=int, default=0, help="log every X updates steps. It will be effective if its value > 0. If value == 0, log at the end of every epoch. If value < 0, do not log at all. Newer version's updates: if value==0, log at the end of every epoch. If value < 0, only log once at the end.")
    parser.add_argument('--save_steps', type=int, default=0, help="save checkpoint every X updates steps. It will be effective if its value > 0. If value == 0, save at the end of every epoch. If value < 0, do not save at all. Newer version's updates: if value==0, save at the end of every epoch. If value < 0, only save once at the end.")
    parser.add_argument('--self_train_epochs', type=float, default=1, help='self training epochs')
    parser.add_argument('--update_interval', type=int, default=50, help='self training update interval')
    parser.add_argument('--early_stop', action='store_true', help='whether or not to enable early stop of self-training')


    # output related hyperparameters
    parser.add_argument('--output_dir', default='results/', help='output directory')
    
    # evaluation related hyperparameters
    parser.add_argument('--num_buffer_examples', type=int, default=30, help='If set to deterministic, first this number of test examples of each relation will be saved for few-shot settings and not for evaluation usage.') 
    
    
    # hyperparams for running snowball
    parser.add_argument('--num_init_pos_examples', type=int, default=30, help='initial positive examples to acquire per relation')
    parser.add_argument('--num_init_neg_examples', type=int, default=60, help='initial negative examples to acquire per relation (including both using LLM and random sampling)')
    parser.add_argument('--run_neg_init_gen', action='store_true', help='whether or not to generate negative relation examples initially using LLM')
    parser.add_argument('--num_init_neg_rels_to_generate', type=int, default=5, help='initial negative relations to generate for each positive relation using LLM')
    parser.add_argument('--num_init_neg_examples_to_generate', type=int, default=30, help='initial negative examples to generate for each positive relation using LLM')
    parser.add_argument('--num_follow_pos_examples', type=int, default=30, help='followup positive examples to further acquire per relation')
    parser.add_argument('--num_follow_neg_examples', type=int, default=60, help='followup negative examples to further acquire per relation')
    parser.add_argument('--run_neg_follow_gen', action='store_true', help='whether or not to generate negative relation examples followup using LLM')
    parser.add_argument('--num_follow_neg_rels_to_generate', type=int, default=5, help='followup negative relations to generate for each positive relation using LLM')
    parser.add_argument('--num_follow_neg_examples_to_generate', type=int, default=30, help='followup negative examples to generate for each positive relation using LLM')
    
    
    parser.add_argument('--run_snowball', action='store_true', help='whether or not to conduct snowball steps. Here "snowball" means inference over unlabeled corpus.')
    parser.add_argument("--negative_sampling_upper_bound", default=1.0, type=float, help="upper bound of the predicted probability of being positive under which negative sampling will be conducted.")
    parser.add_argument('--save_epochs', type=int, nargs='+', default=[2, 9, 4], help='epochs to save the ckpts and evaluation results.')
    parser.add_argument('--choose_by_elbow', action='store_true', help='selecting trained ckpts for test evaluation. If set to True, will select based on the elbow point of training loss curve. If set to False, will use the last element epoch number in args.save_epochs list. (This argument is not related to the main experiments we run. Eventually, the model checkpoint will be chosen based on the performance on the dev set.)')
    parser.add_argument('--run_neg_rel_gen', action='store_true', help='whether or not to generate negative relation definitions using LLM')
    parser.add_argument('--logging_epochs', type=int, nargs='+', default=[4], help='epochs to save the evaluation results.')

    
    # GPT hyperparams
    parser.add_argument("--temperature", default=1.0, type=float, help="general temperature of calling GPT")
    parser.add_argument("--def_gen_temperature", default=1.0, type=float, help="definition generation temperature of calling GPT")
    parser.add_argument("--neg_ex_gen_temperature", default=1.0, type=float, help="negative example generation temperature of calling GPT")
    parser.add_argument('--run_LLM_json_parser_ex', action='store_true', help='whether or not to use GPT to parse the example generations into json format')
    parser.add_argument('--run_LLM_json_parser_def', action='store_true', help='whether or not to use GPT to parse the neg def generations into json format')
    parser.add_argument('--llm_model_ckpt_parser', type=str, default='gpt-4o-mini-2024-07-18', help='model ckpt for calling large language models as json parsers')
    
    
    # parsing and setting the args for model, dataloader and trainer
    args = parser.parse_args()
    args.num_gpus = torch.cuda.device_count()
    args.train_batch_size_per_device = args.train_batch_size
    args.eval_batch_size_per_device = args.eval_batch_size
    args.unlabel_infer_batch_size_per_device = args.unlabel_infer_batch_size
    args.train_batch_size = args.train_batch_size * max(1, args.num_gpus)
    args.eval_batch_size = args.eval_batch_size * max(1, args.num_gpus)
    args.unlabel_infer_batch_size = args.unlabel_infer_batch_size * max(1, args.num_gpus)
    args.num_cpus = min(50, cpu_count() - 1) if cpu_count() > 1 else 1
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print('===All of the running args as follows: ===')
    print(args)
    print('\n')

    dataloader = REDataLoader(args=args)
    model_trainer = ModelTrainer(args=args, model=None, dataloader=dataloader)

    if args.run_type == 'initial':
        model_trainer.run_snowball(K=5, backup=args.num_buffer_examples, test_div='test', unlabeled_corpus_div='distant', save_suffix=None, deterministic=True)
    elif args.run_type == 'followup':
        model_trainer.run_snowball_iterative_main_v1(test_div='test', unlabeled_corpus_div='distant', save_suffix=None, deterministic=True, iterative_version='v0')
    
    
    
if __name__ == '__main__':
    main()