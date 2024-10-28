from utils import get_SBERT_embedding
import torch
import numpy as np
import random
import matplotlib.pyplot as plt 
from spacy import displacy
from sklearn.manifold import TSNE
from copy import deepcopy

# # https://spacy.io/usage/processing-pipelines#_title for more info on spacy nlp.pipe
# spacy.prefer_gpu()
# # nlp = spacy.load("en_core_web_lg")
# nlp = spacy.load("en_core_web_trf")
# # Get the number of available cores/processors
# n_cores = multiprocessing.cpu_count()



def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def is_ascending(array):
    sorted_array = np.sort(array)
    return np.array_equal(array, sorted_array)


def find_st_ed_tokens(doc, word_list, word_st_id, word_ed_id):
    """
    Find the inclusive start and end token id that enclose the inclusive span of the words denoted by word_st_id and word_ed_id

    Args:
        doc (spacy parsed document): obtained from spacy.load("en_core_web_lg")(" ".join(word_list))
        word_list (List[str]): a list of strings, each string is a word
        word_st_id (int): inclusive index of the start word in the word list
        word_ed_id (int): inclusive index of the end word in the word list

    Returns:
        tuple: 
            - int: start idx of the token in doc that includes the start word denoted by word_st_id
            - int: inclusive end idx of the token in the doc that includes the end word denoted by word_ed_id
    """
    # first find the start char id of the word_st_id
    if word_st_id == 0:
        word_st_char_id = 0
    else:
        word_st_char_id = len(" ".join(word_list[:word_st_id])) + 1


    # then find the end char id of the word_ed_id
    if word_ed_id == 0:
        word_ed_char_id = len(word_list[word_ed_id]) - 1
    else:
        word_ed_char_id = len(" ".join(word_list[:word_ed_id + 1])) - 1

    token_st_id, token_ed_id = None, None
    for tok_id, tok in enumerate(doc):
        tok_char_st, tok_char_ed = tok.idx, tok.idx + len(tok.text) - 1
        if tok_char_st <= word_st_char_id <= tok_char_ed:
            token_st_id = tok_id
        if tok_char_st <= word_ed_char_id <= tok_char_ed:
            token_ed_id = tok_id
    
    return token_st_id, token_ed_id


# trace dependency path starting from each span's root, trace back to LCA
def trace_to_ancestor(token, ancestor_index):
    """ 
    Return a list of tokens from input token (exclusive) to ancestor_index (exclusive) from the dependency parse tree

    Args:
        token: a spacy token
        ancestor_index: index of the ancestor token of argument `token`
    Returns:
        a list of tokens from input `token` (exclusive) to `ancestor_index` token (exclusive) from the dependency parse tree
    """
    path = [] # from input token (exclusive) to ancestor_index (exclusive)
    while token.i != ancestor_index:
        path.append(token)
        token = token.head
    
    if len(path) > 0: path = path[1:] # exlude the initial token index
    return path


def remove_included_tokens(tokens_to_clean, token_span):
    """
    Remove toekns which are in the token_span from tokens_to_clean. 
    Unchange the token order in tokens_to_clean (if it's list).

    Args: 
        tokens_to_clean: a single spacy token or a list of spacy tokens
        token_span: a spacy span or a list of spacy tokens
    Returns:
        a list of tokens from tokens_to_clean that do not occur in token_span
    """

    if not isinstance(tokens_to_clean, list): # tokens_to_clean is a single token
        for token in token_span:
            if token.i == tokens_to_clean.i: return []
        return [tokens_to_clean]
    else:
        tokens_cleaned = []
        for token_to_clean in tokens_to_clean:
            to_remove = False
            for token in token_span:
                if token.i == token_to_clean.i: to_remove = True
            if not to_remove: tokens_cleaned.append(token_to_clean)
        return tokens_cleaned


def get_shortest_dep_path(word_list, ht_word_spans):
    """ 
    Utilize spacy to obtain the minimal shortest dependency path from head entity to tail entity. 

    Args: 
        word_list: list of words (strings) representing one sentence
        ht_word_spans (note all indices below are inclusive): 
            [
                [head entity start word index, head entity end word index], 
                [tail entity start word index, tail entity end word index],
            ]
        
    Returns:
        doc: spacy doc by inputing " ".join(word_list)
        h_st_token_id: [inclusive] start token index (of doc) of head entity
        h_ed_token_id: [inclusive] end token index (of doc) of head entity
        t_st_token_id: [inclusive] start token index (of doc) of tail entity
        t_ed_token_id: [inclusive] end token index (of doc) of tail entity
        h_doc_span: spacy span of head entity
        t_doc_span: spacy span of tail entity
        h_doc_span_path: list of spacy tokens starting from h_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        t_doc_span_path: list of spacy tokens starting from t_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        dependency_path: list of spacy tokens starting from h_doc_span_root token (excluded) to t_doc_span_root token (excluded) (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    """
    doc = nlp(" ".join(word_list))

    h_st_token_id, h_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[0][0], word_ed_id=ht_word_spans[0][1])
    t_st_token_id, t_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[1][0], word_ed_id=ht_word_spans[1][1])

    # ideally, h_doc_span and t_doc_span do not intersect
    h_doc_span = doc[h_st_token_id: h_ed_token_id + 1]
    t_doc_span = doc[t_st_token_id: t_ed_token_id + 1]

    h_t_tokens_between = doc[h_ed_token_id + 1: t_st_token_id] if h_ed_token_id <= t_st_token_id else doc[t_ed_token_id + 1: h_st_token_id]

    # find root token of each span
    h_doc_span_root = h_doc_span.root # highest level node (token) among h_doc_span tokens
    t_doc_span_root = t_doc_span.root # highest level node (token) among t_doc_span tokens

    # find lowest common ancestor (LCA)
    lca = doc.get_lca_matrix() 
    common_ancestor = lca[h_doc_span_root.i, t_doc_span_root.i] # token index
    common_ancestor_token = doc[common_ancestor] # token


    # trace dependency path starting from each span's root token, trace back to LCA token
    h_doc_span_path = trace_to_ancestor(h_doc_span_root, common_ancestor) # tokens from h_doc_span_root (excluded) to common ancestor token (excluded)
    t_doc_span_path = trace_to_ancestor(t_doc_span_root, common_ancestor) # tokens from t_doc_span_root (excluded) to common ancestor token (excluded)



    # combine path (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    dependency_path = remove_included_tokens(tokens_to_clean=h_doc_span_path, token_span=t_doc_span) + remove_included_tokens(tokens_to_clean=common_ancestor_token, token_span=list(h_doc_span) + list(t_doc_span)) + remove_included_tokens(tokens_to_clean=t_doc_span_path[::-1], token_span=h_doc_span)


    # # output dependency path
    # for token in dependency_path: print(token.text, token.dep_, token.head.text)

    return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path


def is_token_in_span(token, token_span):
    """
    Check if token is included in token_span.

    Args:
        token: spacy token
        token_span: spacy span or a list of spacy tokens
    Returns:
        True if token is included in the token_span. Otherwise, False.
    """
    for tok in token_span:
        if token.i == tok.i: return True
    return False


def linearize_ent_dep_path(doc, h_doc_span, t_doc_span, dependency_path=None, add_tags=True, add_h_t=True):
    """ 
    Args:
        doc: spacy doc of whole sentence
        h_doc_span: spacy span of the head entity
        t_doc_span: spacy span of the tail entity
        [Optional] dependency_path: list of spacy tokens. If this arg is None, will print all the tokens except for h and t tokens. 
    Returns: 
        a list of token strings with entity tags before and after each entity for visualization
    """

    ordered_tokens_to_print = []

    if dependency_path is None: dependency_path = [tok for tok in doc if not (is_token_in_span(token=tok, token_span=h_doc_span) or is_token_in_span(token=tok, token_span=t_doc_span))]

    for tok in doc:
        if add_tags:
            if tok.i == h_doc_span[0].i: ordered_tokens_to_print.append('<ENT0>')
            if tok.i == t_doc_span[0].i: ordered_tokens_to_print.append('<ENT1>')
        if add_h_t:
            if is_token_in_span(token=tok, token_span=h_doc_span) or is_token_in_span(token=tok, token_span=t_doc_span) or is_token_in_span(token=tok, token_span=dependency_path):
                ordered_tokens_to_print.append(tok.text)
        else:
            if is_token_in_span(token=tok, token_span=dependency_path):
                ordered_tokens_to_print.append(tok.text)
        if add_tags:
            if tok.i == h_doc_span[-1].i: ordered_tokens_to_print.append('</ENT0>')
            if tok.i == t_doc_span[-1].i: ordered_tokens_to_print.append('</ENT1>')

    return ordered_tokens_to_print


def linearize_ent_dep_path_w_mask(doc, h_doc_span, t_doc_span, dependency_path=None, mask_h_t=False, mask_token='<mask>', tag_h_t=False):
    ordered_tokens_to_print = []
    if dependency_path is None: dependency_path = [tok for tok in doc if not (is_token_in_span(token=tok, token_span=h_doc_span) or is_token_in_span(token=tok, token_span=t_doc_span))]

    for tok_i, tok in enumerate(doc):

        if mask_h_t:
            if not (is_token_in_span(token=tok, token_span=dependency_path)) and (tok_i == 0 or ordered_tokens_to_print[-1] != mask_token):
                ordered_tokens_to_print.append(mask_token)
                continue
            if is_token_in_span(token=tok, token_span=dependency_path):
                ordered_tokens_to_print.append(tok.text)
        else:
            if not (is_token_in_span(token=tok, token_span=h_doc_span) or is_token_in_span(token=tok, token_span=t_doc_span) or is_token_in_span(token=tok, token_span=dependency_path)) and (tok_i == 0 or ordered_tokens_to_print[-1] != mask_token):
                ordered_tokens_to_print.append(mask_token)
                continue
            if tag_h_t: 
                if tok.i == h_doc_span[0].i: ordered_tokens_to_print.append('<ENT0>')
                if tok.i == t_doc_span[0].i: ordered_tokens_to_print.append('<ENT1>')
            if is_token_in_span(token=tok, token_span=h_doc_span) or is_token_in_span(token=tok, token_span=t_doc_span) or is_token_in_span(token=tok, token_span=dependency_path):
                ordered_tokens_to_print.append(tok.text)
            if tag_h_t:
                if tok.i == h_doc_span[-1].i: ordered_tokens_to_print.append('</ENT0>')
                if tok.i == t_doc_span[-1].i: ordered_tokens_to_print.append('</ENT1>')

    return ordered_tokens_to_print


def get_shortest_dep_path_v1(word_list, ht_word_spans, doc_given=None):
    """ 
    v0: Utilize spacy to obtain the minimal shortest dependency path from head entity to tail entity. 
    v1: for each token in the dependency path, include the immediate child token as well.

    Args: 
        word_list: list of words (strings) representing one sentence
        ht_word_spans (note all indices below are inclusive): 
            [
                [head entity start word index, head entity end word index], 
                [tail entity start word index, tail entity end word index],
            ]
        
    Returns:
        doc: spacy doc by inputing " ".join(word_list)
        h_st_token_id: [inclusive] start token index (of doc) of head entity
        h_ed_token_id: [inclusive] end token index (of doc) of head entity
        t_st_token_id: [inclusive] start token index (of doc) of tail entity
        t_ed_token_id: [inclusive] end token index (of doc) of tail entity
        h_doc_span: spacy span of head entity
        t_doc_span: spacy span of tail entity
        h_doc_span_path: list of spacy tokens starting from h_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        t_doc_span_path: list of spacy tokens starting from t_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        dependency_path: list of spacy tokens starting from h_doc_span_root token (excluded) to t_doc_span_root token (excluded) (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    """
    if doc_given is None:
        doc = nlp(" ".join(word_list))
    else: 
        doc = doc_given

    h_st_token_id, h_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[0][0], word_ed_id=ht_word_spans[0][1])
    t_st_token_id, t_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[1][0], word_ed_id=ht_word_spans[1][1])

    # ideally, h_doc_span and t_doc_span do not intersect
    h_doc_span = doc[h_st_token_id: h_ed_token_id + 1]
    t_doc_span = doc[t_st_token_id: t_ed_token_id + 1]

    h_t_tokens_between = doc[h_ed_token_id + 1: t_st_token_id] if h_ed_token_id <= t_st_token_id else doc[t_ed_token_id + 1: h_st_token_id]

    # find root token of each span
    h_doc_span_root = h_doc_span.root # hgihest level node (token) among h_doc_span tokens
    t_doc_span_root = t_doc_span.root # highest level node (token) among t_doc_span tokens

    # find lowest common ancestor (LCA)
    lca = doc.get_lca_matrix() 
    common_ancestor = lca[h_doc_span_root.i, t_doc_span_root.i] # token index
    common_ancestor_token = doc[common_ancestor] # token

    if common_ancestor == -1:
        return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, None, None, None
    

    # trace dependency path starting from each span's root token, trace back to LCA token
    h_doc_span_path = trace_to_ancestor(h_doc_span_root, common_ancestor) # tokens from h_doc_span_root (excluded) to common ancestor token (excluded)
    t_doc_span_path = trace_to_ancestor(t_doc_span_root, common_ancestor) # tokens from t_doc_span_root (excluded) to common ancestor token (excluded)



    # combine path (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    dependency_path = remove_included_tokens(tokens_to_clean=h_doc_span_path, token_span=t_doc_span) + remove_included_tokens(tokens_to_clean=common_ancestor_token, token_span=list(h_doc_span) + list(t_doc_span)) + remove_included_tokens(tokens_to_clean=t_doc_span_path[::-1], token_span=h_doc_span)


    # function to include immediate children of a token
    def include_children(token, included_tokens):
        for child in token.children:
            if child not in included_tokens: included_tokens.add(child)
    # construct more informative dependency paths
    extended_tokens = set()
    for tok in dependency_path:
        extended_tokens.add(tok)
        include_children(token=tok, included_tokens=extended_tokens)
    dependency_path = list(extended_tokens)
    # remove tokens that already occurred in the head entity or tail entity
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=h_doc_span)
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=t_doc_span)

    return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path


def get_shortest_dep_path_v2(word_list, ht_word_spans, doc_given=None):
    """ 
    v0: Utilize spacy to obtain the minimal shortest dependency path from head entity to tail entity. 
    v1: for each token in the dependency path, include the immediate child token as well.
    v2: for each token in the dependency path and each token in the head entity and tail entity, include the immediate child token as well.

    Args: 
        word_list: list of words (strings) representing one sentence
        ht_word_spans (note all indices below are inclusive): 
            [
                [head entity start word index, head entity end word index], 
                [tail entity start word index, tail entity end word index],
            ]
        
    Returns:
        doc: spacy doc by inputing " ".join(word_list)
        h_st_token_id: [inclusive] start token index (of doc) of head entity
        h_ed_token_id: [inclusive] end token index (of doc) of head entity
        t_st_token_id: [inclusive] start token index (of doc) of tail entity
        t_ed_token_id: [inclusive] end token index (of doc) of tail entity
        h_doc_span: spacy span of head entity
        t_doc_span: spacy span of tail entity
        h_doc_span_path: list of spacy tokens starting from h_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        t_doc_span_path: list of spacy tokens starting from t_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        dependency_path: list of spacy tokens starting from h_doc_span_root token (excluded) to t_doc_span_root token (excluded) (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    """
    if doc_given is None:
        doc = nlp(" ".join(word_list))
    else: 
        doc = doc_given

    h_st_token_id, h_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[0][0], word_ed_id=ht_word_spans[0][1])
    t_st_token_id, t_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[1][0], word_ed_id=ht_word_spans[1][1])

    # ideally, h_doc_span and t_doc_span do not intersect
    h_doc_span = doc[h_st_token_id: h_ed_token_id + 1]
    t_doc_span = doc[t_st_token_id: t_ed_token_id + 1]

    h_t_tokens_between = doc[h_ed_token_id + 1: t_st_token_id] if h_ed_token_id <= t_st_token_id else doc[t_ed_token_id + 1: h_st_token_id]

    # find root token of each span
    h_doc_span_root = h_doc_span.root # hgihest level node (token) among h_doc_span tokens
    t_doc_span_root = t_doc_span.root # highest level node (token) among t_doc_span tokens

    # find lowest common ancestor (LCA)
    lca = doc.get_lca_matrix() 
    common_ancestor = lca[h_doc_span_root.i, t_doc_span_root.i] # token index
    common_ancestor_token = doc[common_ancestor] # token

    if common_ancestor == -1:
        return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, None, None, None
    

    # trace dependency path starting from each span's root token, trace back to LCA token
    h_doc_span_path = trace_to_ancestor(h_doc_span_root, common_ancestor) # tokens from h_doc_span_root (excluded) to common ancestor token (excluded)
    t_doc_span_path = trace_to_ancestor(t_doc_span_root, common_ancestor) # tokens from t_doc_span_root (excluded) to common ancestor token (excluded)



    # combine path (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    dependency_path = remove_included_tokens(tokens_to_clean=h_doc_span_path, token_span=t_doc_span) + remove_included_tokens(tokens_to_clean=common_ancestor_token, token_span=list(h_doc_span) + list(t_doc_span)) + remove_included_tokens(tokens_to_clean=t_doc_span_path[::-1], token_span=h_doc_span)


    # function to include immediate children of a token
    def include_children(token, included_tokens):
        for child in token.children:
            if child not in included_tokens: included_tokens.add(child)
    # construct more informative dependency paths
    extended_tokens = set()
    for tok in dependency_path:
        extended_tokens.add(tok)
        include_children(token=tok, included_tokens=extended_tokens)
    for tok in h_doc_span: include_children(token=tok, included_tokens=extended_tokens)
    for tok in t_doc_span: include_children(token=tok, included_tokens=extended_tokens)
    dependency_path = list(extended_tokens)
    # remove tokens that are already included by the head entity and tail entity
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=h_doc_span)
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=t_doc_span)

    return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path


def get_shortest_dep_path_v3(word_list, ht_word_spans, doc_given=None):
    """ 
    v0: Utilize spacy to obtain the minimal shortest dependency path from head entity to tail entity. 
    v1: for each token in the dependency path, include the immediate child token as well.
    v2: for each token in the dependency path and each token in the head entity and tail entity, include the immediate child token as well.
    v3: for each token in the head entity and tail entity, include the immediate child token. For each verb token in the dependency path, include its related tokens as well.

    Args: 
        word_list: list of words (strings) representing one sentence
        ht_word_spans (note all indices below are inclusive): 
            [
                [head entity start word index, head entity end word index], 
                [tail entity start word index, tail entity end word index],
            ]
    
    Returns:
        doc: spacy doc by inputing " ".join(word_list)
        h_st_token_id: [inclusive] start token index (of doc) of head entity
        h_ed_token_id: [inclusive] end token index (of doc) of head entity
        t_st_token_id: [inclusive] start token index (of doc) of tail entity
        t_ed_token_id: [inclusive] end token index (of doc) of tail entity
        h_doc_span: spacy span of head entity
        t_doc_span: spacy span of tail entity
        h_doc_span_path: list of spacy tokens starting from h_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        t_doc_span_path: list of spacy tokens starting from t_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        dependency_path: list of spacy tokens starting from h_doc_span_root token (excluded) to t_doc_span_root token (excluded) (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    """
    if doc_given is None:
        doc = nlp(" ".join(word_list))
    else: 
        doc = doc_given

    h_st_token_id, h_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[0][0], word_ed_id=ht_word_spans[0][1])
    t_st_token_id, t_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[1][0], word_ed_id=ht_word_spans[1][1])

    # ideally, h_doc_span and t_doc_span do not intersect
    h_doc_span = doc[h_st_token_id: h_ed_token_id + 1]
    t_doc_span = doc[t_st_token_id: t_ed_token_id + 1]

    h_t_tokens_between = doc[h_ed_token_id + 1: t_st_token_id] if h_ed_token_id <= t_st_token_id else doc[t_ed_token_id + 1: h_st_token_id]

    # find root token of each span
    h_doc_span_root = h_doc_span.root # hgihest level node (token) among h_doc_span tokens
    t_doc_span_root = t_doc_span.root # highest level node (token) among t_doc_span tokens

    # find lowest common ancestor (LCA)
    lca = doc.get_lca_matrix() 
    common_ancestor = lca[h_doc_span_root.i, t_doc_span_root.i] # token index
    common_ancestor_token = doc[common_ancestor] # token

    if common_ancestor == -1:
        return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, None, None, None
    

    # trace dependency path starting from each span's root token, trace back to LCA token
    h_doc_span_path = trace_to_ancestor(h_doc_span_root, common_ancestor) # tokens from h_doc_span_root (excluded) to common ancestor token (excluded)
    t_doc_span_path = trace_to_ancestor(t_doc_span_root, common_ancestor) # tokens from t_doc_span_root (excluded) to common ancestor token (excluded)



    # combine path (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    dependency_path = remove_included_tokens(tokens_to_clean=h_doc_span_path, token_span=t_doc_span) + remove_included_tokens(tokens_to_clean=common_ancestor_token, token_span=list(h_doc_span) + list(t_doc_span)) + remove_included_tokens(tokens_to_clean=t_doc_span_path[::-1], token_span=h_doc_span)


    # function to include immediate children of a token
    def include_children(token, included_tokens):
        for child in token.children:
            if child not in included_tokens: included_tokens.add(child)
    def include_verb_related_tokens(token, included_tokens):
        for child in token.children:
            if token.pos_== 'VERB' and (child.dep_ in {'dobj', 'prep', 'agent', 'aux', 'auxpass'}):
                included_tokens.add(child)
    # construct more informative dependency paths
    extended_tokens = set()
    for tok in h_doc_span: include_children(token=tok, included_tokens=extended_tokens)
    for tok in t_doc_span: include_children(token=tok, included_tokens=extended_tokens)
    for tok in dependency_path: 
        extended_tokens.add(tok)
        include_verb_related_tokens(token=tok, included_tokens=extended_tokens)
    dependency_path = list(extended_tokens)

    # remove tokens that are already included by the head entity and tail entity
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=h_doc_span)
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=t_doc_span)

    return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path


def get_shortest_dep_path_v4(word_list, ht_word_spans, doc_given=None):
    """ 
    v0: Utilize spacy to obtain the minimal shortest dependency path from head entity to tail entity. 
    v1: for each token in the dependency path, include the immediate child token as well.
    v2: for each token in the dependency path and each token in the head entity and tail entity, include the immediate child token as well.
    v3: for each token in the head entity and tail entity, include the immediate child token. For each verb token in the dependency path, include its related tokens as well.
    v4: for each token in the head entity and tail entity, include the immediate child token.

    Args: 
        word_list: list of words (strings) representing one sentence
        ht_word_spans (note all indices below are inclusive): 
            [
                [head entity start word index, head entity end word index], 
                [tail entity start word index, tail entity end word index],
            ]
        
    Returns:
        doc: spacy doc by inputing " ".join(word_list)
        h_st_token_id: [inclusive] start token index (of doc) of head entity
        h_ed_token_id: [inclusive] end token index (of doc) of head entity
        t_st_token_id: [inclusive] start token index (of doc) of tail entity
        t_ed_token_id: [inclusive] end token index (of doc) of tail entity
        h_doc_span: spacy span of head entity
        t_doc_span: spacy span of tail entity
        h_doc_span_path: list of spacy tokens starting from h_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        t_doc_span_path: list of spacy tokens starting from t_doc_span_root token (excluded) to lowest common ancestor token (excluded)
        dependency_path: list of spacy tokens starting from h_doc_span_root token (excluded) to t_doc_span_root token (excluded) (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    """
    if doc_given is None:
        doc = nlp(" ".join(word_list))
    else: 
        doc = doc_given

    h_st_token_id, h_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[0][0], word_ed_id=ht_word_spans[0][1])
    t_st_token_id, t_ed_token_id = find_st_ed_tokens(doc=doc, word_list=word_list, word_st_id=ht_word_spans[1][0], word_ed_id=ht_word_spans[1][1])

    # ideally, h_doc_span and t_doc_span do not intersect
    h_doc_span = doc[h_st_token_id: h_ed_token_id + 1]
    t_doc_span = doc[t_st_token_id: t_ed_token_id + 1]

    h_t_tokens_between = doc[h_ed_token_id + 1: t_st_token_id] if h_ed_token_id <= t_st_token_id else doc[t_ed_token_id + 1: h_st_token_id]

    # find root token of each span
    h_doc_span_root = h_doc_span.root # hgihest level node (token) among h_doc_span tokens
    t_doc_span_root = t_doc_span.root # highest level node (token) among t_doc_span tokens

    # find lowest common ancestor (LCA)
    lca = doc.get_lca_matrix() 
    common_ancestor = lca[h_doc_span_root.i, t_doc_span_root.i] # token index
    common_ancestor_token = doc[common_ancestor] # token

    if common_ancestor == -1:
        return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, None, None, None

    # trace dependency path starting from each span's root token, trace back to LCA token
    h_doc_span_path = trace_to_ancestor(h_doc_span_root, common_ancestor) # tokens from h_doc_span_root (excluded) to common ancestor token (excluded)
    t_doc_span_path = trace_to_ancestor(t_doc_span_root, common_ancestor) # tokens from t_doc_span_root (excluded) to common ancestor token (excluded)



    # combine path (token order will be from h_doc_span_root to t_doc_span_root) (dependnecy_path is guaranteed to not have tokens from h_doc_span and t_doc_span) 
    dependency_path = remove_included_tokens(tokens_to_clean=h_doc_span_path, token_span=t_doc_span) + remove_included_tokens(tokens_to_clean=common_ancestor_token, token_span=list(h_doc_span) + list(t_doc_span)) + remove_included_tokens(tokens_to_clean=t_doc_span_path[::-1], token_span=h_doc_span)


    # function to include immediate children of a token
    def include_children(token, included_tokens):
        for child in token.children:
            if child not in included_tokens: included_tokens.add(child)
    def include_verb_related_tokens(token, included_tokens):
        for child in token.children:
            if token.pos_== 'VERB' and (child.dep_ in {'dobj', 'prep', 'agent', 'aux', 'auxpass'}):
                included_tokens.add(child)
    # construct more informative dependency paths
    extended_tokens = set()
    for tok in h_doc_span: include_children(token=tok, included_tokens=extended_tokens)
    for tok in t_doc_span: include_children(token=tok, included_tokens=extended_tokens)
    for tok in dependency_path: 
        extended_tokens.add(tok)
        # include_verb_related_tokens(token=tok, included_tokens=extended_tokens)
    dependency_path = list(extended_tokens)

    # remove tokens that are already included by the head entity and tail entity
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=h_doc_span)
    dependency_path = remove_included_tokens(tokens_to_clean=dependency_path, token_span=t_doc_span)

    return doc, h_st_token_id, h_ed_token_id, t_st_token_id, t_ed_token_id, h_doc_span, t_doc_span, h_doc_span_path, t_doc_span_path, dependency_path



def get_KMeans_clusters(data_embeddings, seed=0, n_clusters=10):
    """
    Args:
        data_embeddings: numpy array of shape (n_samples, n_features) representing the embeddings for data samples
    
    Returns:
        clustering_model.cluster_centers_: ndarray of shape (n_clusters, n_features), Coordinates of cluster centers.
        cluster_assignment: ndarray of shape (n_samples,), Labels of each point
        clustered_data_indices: list of n_clusters lists, each list corresponds to one cluster i and contains sample ids that are assigned to cluster i 
        clustered_dists: list of n_clusters lists, each list corresponds to one cluster i and contains distance of the assigned data sample to cluster i 
        dist: ndarray of shape (n_samples, n_clusters), transform data_embeddings to cluster-distance space where each dimension is the distance to the cluster centers
    """
    from sklearn.cluster import KMeans
    fix_seed(seed=seed)

    data_embeddings_copy = deepcopy(data_embeddings)

    clustering_model = KMeans(n_clusters=n_clusters, random_state=seed, init='k-means++', n_init='auto')
    clustering_model.fit(data_embeddings_copy)
    cluster_assignment = clustering_model.labels_ # (n_samples,)

    clustered_data_indices = [[] for i in range(n_clusters)] # each list corresponds to one cluster i and contains sample ids that are assigned to cluster i 
    clustered_dists = [[] for i in range(n_clusters)] # each list corresponds to one cluster i and contains distance of the assigned data sample to cluster i 

    dist = clustering_model.transform(data_embeddings_copy) # (n_samples, n_clusters), transform data_embeddings to cluster-distance space where each dimension is the distance to the cluster centers
    for data_id, cluster_id in enumerate(cluster_assignment):
        clustered_data_indices[cluster_id].append(data_id)
        clustered_dists[cluster_id].append(dist[data_id][cluster_id])

    return clustering_model.cluster_centers_, cluster_assignment, clustered_data_indices, clustered_dists, dist


def get_HDBSCAN_clusters(data_embeddings, seed=0, min_cluster_size=5, min_samples=None, metric='euclidean', n_jobs=-1, cluster_selection_method='eom', store_centers="centroid"):
    """ 
    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html 

    Args: 
        data_embeddings: numpy array of shape (n_samples, n_features) representing the embeddings for data samples
        min_cluster_size: minimum number of samples in a group for that group to be considered a cluster; groupings smaller than this size will be left as noise
        min_samples: number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself. When None, defaults to min_cluster_size
        metric: cosine similarity does not follow triangle inequality, so we use angular similarity instead which requires the data_embeddings to be L2 normalized and the metric to be euclidean (need to set normalize to True when using sentence transformer to encode which results in L2 normalization). See https://github.com/scikit-learn-contrib/hdbscan/issues/69.  
        n_jobs: number of processors/codes to use for computing
        cluster_selection_method:  Excess of Mass ("eom") algorithm to find the most persistent clusters. Alternatively you can instead select the clusters at the leaves of the tree ("leaf") this provides the most fine grained and homogeneous clusters.
        store_centers: Which, if any, cluster centers to compute and store. The options are: None | "centroid" | "medoid" | "both" (we require not using "both" for convenience)
    
    Returns:
        cluster_assignment: (n_samples,), cluster assignment labels for each data sample, samples clustered as noise will have label=-1
        data_cluster_assignment_probs: numpy array of (n_samples,), strength with which each sample is a member of its assigned cluster
        n_clusters: number of clusters (noise do not belong to any cluster)
        clustered_data_indices: list of n_clusters + 1 lists, each list corresponds to one cluster i and contains sample ids that are assigned to cluster i. The last list contains the noise data point ids
        clustered_dists: list of n_clusters + 1 lists, each list corresponds to one cluster i and contains assignment probabilities of the assigned data sample to cluster i. The last list contains the noise data point's assignment probabilities 
        cluster_centroids: None if store_centers=None, else (n_clusters, n_features)
        
    """
    from sklearn.cluster import HDBSCAN
    fix_seed(seed=seed)

    data_embeddings_copy = deepcopy(data_embeddings)

    clustering_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, n_jobs=n_jobs, cluster_selection_method=cluster_selection_method, store_centers=store_centers)
    clustering_model.fit(data_embeddings_copy)
    

    # Noisy samples are given the label -1.
    # Samples with infinite elements (+/- np.inf) are given the label -2.
    # Samples with missing data are given the label -3, even if they also have infinite elements.
    cluster_assignment = clustering_model.labels_ # (n_samples,)

    # Noisy samples have probability zero.
    # Samples with infinite elements (+/- np.inf) have probability 0.
    # Samples with missing data have probability np.nan.
    data_cluster_assignment_probs = clustering_model.probabilities_ # (n_samples,), strength with which each sample is a member of its assigned cluster

    n_clusters = len(set(cluster_assignment[cluster_assignment >= 0])) # number of clusters (noise do not belong to any cluster)
    clustered_data_indices = [[] for i in range(n_clusters + 1)] # each list corresponds to one cluster i and contains sample ids that are assigned to cluster i. The last list contains the noise data point ids
    clustered_dists = [[] for i in range(n_clusters + 1)] # each list corresponds to one cluster i and contains assignment probabilities of the assigned data sample to cluster i. The last list contains the noise data point's assignment probabilities 

    for data_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id < -1: continue # ignore infinite values and missing values
        clustered_data_indices[cluster_id].append(data_id)
        clustered_dists[cluster_id].append(data_cluster_assignment_probs[data_id])


    # None or ndarray of shape (n_clusters, n_features)
    #   Note that n_clusters only counts non-outlier clusters. That is to say, the -1, -2, -3 labels for the outlier clusters are excluded
    cluster_centroids = None 
    if store_centers == "centroid": cluster_centroids = clustering_model.centroids_
    if store_centers == "medoid": cluster_centroids = clustering_model.medoids_
    

    return cluster_assignment, data_cluster_assignment_probs, n_clusters, clustered_data_indices, clustered_dists, cluster_centroids 







def cluster_plot(X, labels, centroid_X=None, probabilities=None, parameters=None, ground_truth=False, ax=None, add_text=False):
    """ 
    Visualization codes adapted from scikit-learn. 

    Args:
        X: numpy array of (n_samples, 2)
        labels [Optional, default to one array of (n_samples, )]: numpy array of (n_samples, ) with -1 specially for noise. Different labels will have different point colors.
        centroid_X [Optional]: if set to (n_clusters, 2), will plot the centroids also
        probabilities [Optional, default to one array of (n_samples, )]: array of (n_samples, ), different probability will have different point size.
        parameters [Optional]: dict containing parameter name and values which will appear in the plot title
        ground_truth: True for "True" which appear in the plot title and False for "Estimated" which appear in the plot title
        ax [Optional]: axis
        add_text: whether to add texts beside each data point
    Returns:
        None. Plot! Noise data points are visualized as black crosses.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    # probability of a point belonging to its labeled cluster determines the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )

            # add texts beside each data point if required
            if add_text:
                ax.text(
                    X[ci, 0],
                    X[ci, 1],
                    f"{ci}_{k}",
                    fontdict={'weight': 'bold', 'size': 10}
                )
        
    if centroid_X is not None:
        for k, col in zip(unique_labels, colors):
            if k == -1: continue
            ax.plot(
                centroid_X[k, 0],
                centroid_X[k, 1],
                "*",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

            # add texts beside each data point if required
            if add_text:
                ax.text(
                    centroid_X[k, 0],
                    centroid_X[k, 1],
                    f"CT_{k}",
                    fontdict={'weight': 'bold', 'size': 12}
                )
                
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()


def display_doc(doc, ):
    """
    Visualize the dependency parse tree of the input doc
    """
    print('Token\t\t\tPOS_\t\t\tDEP_:')
    for tok in doc: print(f"{tok}\t\t\t{tok.pos_}\t\t\t{tok.dep_}")

    # customize the visualization
    options = {"compact": False, "color": "green", "distance": 120}
    displacy.render(doc, style="dep", options=options)





def get_tsne_embeddings(data_embeddings, centroid_embeddings=None, seed=0, perplexity=30):
    """ 
    Transform embeddings of high dimension to 2 dimension embeddings for visualization purpose based on t-SNE

    Args:
        data_embeddings: numpy array of (n_samples, n_features)
        centroid_embeddings: numpy array of (n_clusters, n_features)
    Returns:
        data_embeddings_transformed: numpy array of (n_samples, 2)
        centroid_embeddings_transformed: None if centroid_embeddings=None, else numpy array of (n_clusters, 2)
    """
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)

    data_embeddings_copy = deepcopy(data_embeddings)
    if centroid_embeddings is not None: 
        centroid_embeddings_copy = deepcopy(centroid_embeddings)
        all_embeddings = np.concatenate((data_embeddings_copy, centroid_embeddings_copy), axis=0)
    else: all_embeddings = data_embeddings_copy

    all_transformed = tsne.fit_transform(all_embeddings)
    data_embeddings_transformed, centroid_embeddings_transformed = None, None
    if centroid_embeddings is not None:
        data_embeddings_transformed = all_transformed[:data_embeddings.shape[0], :]
        centroid_embeddings_transformed = all_transformed[data_embeddings.shape[0]:, :]
    else:   
        data_embeddings_transformed = all_transformed
    
    return data_embeddings_transformed, centroid_embeddings_transformed




def remove_duplicated_examples(examples):
    examples_wo_duplication = []
    sent_h_t = set()
    for ex in examples:
        sent = " ".join(ex['tokens'])
        h = " ".join([ex['tokens'][i] for i in ex['h'][2][0]])
        t = " ".join([ex['tokens'][i] for i in ex['t'][2][0]])
        ex_sent_h_t = f"{sent}-{h}-{t}"
        if ex_sent_h_t in sent_h_t: continue
        examples_wo_duplication.append(ex)
        sent_h_t.add(ex_sent_h_t)
    return examples_wo_duplication



def remove_duplicated_invalid_examples(examples):
    examples_wo_duplication = []
    sent_h_t = set()
    for ex in examples:
        if len(set(ex['h'][2][0]) & set(ex['t'][2][0])) > 0: continue
        sent = " ".join(ex['tokens'])
        h = " ".join([ex['tokens'][i] for i in ex['h'][2][0]])
        t = " ".join([ex['tokens'][i] for i in ex['t'][2][0]])
        ex_sent_h_t = f"{sent}-{h}-{t}"
        if ex_sent_h_t in sent_h_t: continue
        examples_wo_duplication.append(ex)
        sent_h_t.add(ex_sent_h_t)
    return examples_wo_duplication


def remove_duplicated_invalid_examples_v1(examples, scores):
    assert len(examples) == len(scores)
    examples_wo_duplication = []
    examples_wo_duplication_scores = []
    sent_h_t = set()
    for ex_id, ex in enumerate(examples):
        if len(set(ex['h'][2][0]) & set(ex['t'][2][0])) > 0: continue
        sent = " ".join(ex['tokens'])
        h = " ".join([ex['tokens'][i] for i in ex['h'][2][0]])
        t = " ".join([ex['tokens'][i] for i in ex['t'][2][0]])
        ex_sent_h_t = f"{sent}-{h}-{t}"
        if ex_sent_h_t in sent_h_t: continue
        examples_wo_duplication.append(ex)
        sent_h_t.add(ex_sent_h_t)
        examples_wo_duplication_scores.append(scores[ex_id])
    return examples_wo_duplication, examples_wo_duplication_scores


def select_valid_distinct_examples(ref_examples, buffered_examples, target_num_samples):
    """
    Select target_num_samples distinct examples from buffered_examples (also should all be distinct from ref_examples). Use f"{sent}-{h}-{t}" as the identity of an example.
    """
    selected_examples = []
    sent_h_t = set()
    for ref_ex in ref_examples:
        sent = " ".join(ref_ex['tokens'])
        h = " ".join([ref_ex['tokens'][i] for i in ref_ex['h'][2][0]])
        t = " ".join([ref_ex['tokens'][i] for i in ref_ex['t'][2][0]])
        ex_sent_h_t = f"{sent}-{h}-{t}"
        sent_h_t.add(ex_sent_h_t)
    
    for ex in buffered_examples:
        if len(selected_examples) >= target_num_samples: return selected_examples
        if len(set(ex['h'][2][0]) & set(ex['t'][2][0])) > 0: continue
        sent = " ".join(ex['tokens'])
        h = " ".join([ex['tokens'][i] for i in ex['h'][2][0]])
        t = " ".join([ex['tokens'][i] for i in ex['t'][2][0]])
        ex_sent_h_t = f"{sent}-{h}-{t}"
        if ex_sent_h_t in sent_h_t: continue
        selected_examples.append(ex)
        sent_h_t.add(ex_sent_h_t)
        
    if len(selected_examples) < target_num_samples:
        print(f"Selected valid while distinct examples different from ref_examples are fewer than the target value ({target_num_samples})")
    return selected_examples
    