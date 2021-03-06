import pandas as pd
import numpy as np
import random
import json

from typing import Tuple, List

import os
import os.path
import logging

from nltk import ngrams

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Type Aliases
# class PostRefs(TypedDict):
class PostRefs: 
    post: Tuple[int, int]
    timestamp: int # used as recency, 0 meaning most recent
Perspective = List[PostRefs]


class Conversation:
    """Takes care of the data"""
        
    def __init__(self, global_parameters:dict=None, topic:dict=None):
        self.topic = topic
        self.global_parameters = global_parameters
        self.max_tokens_per_initial_claim = 70
        
        # set up the dataframe
        columns = ['post','peers','perspective','tokens','polarity','salience']
        steps = np.arange(global_parameters['max_t'])
        agents = np.arange(global_parameters['n_agents'])
        steps_agents = [
           steps,
           agents
        ]
        index = pd.MultiIndex.from_product(steps_agents, names=["step", "agent"])
        self.data = pd.DataFrame(np.zeros((global_parameters['max_t']*global_parameters['n_agents'], len(columns))), index=index, columns=columns)
        self.data = self.data.astype(object)
        self.data['perspective']=[[] for i in range(len(self.data))]
        self.data['peers']=[[] for i in range(len(self.data))]
        self.data['tokens']=[[] for i in range(len(self.data))]
        self.data['post']=[None for i in range(len(self.data))]
        

    def contribute(self, contribution=None, agent:int=0, t:int=0, col:str=None):
        self.data.loc[t,agent][col] = contribution

    def get(self, agent:int=0, t:int=0, col:str=None):
        return self.data.loc[t,agent][col]

    def submit_post(self, post=None, agent:int=0, t:int=0):
        self.contribute(contribution=post, agent=agent, t=t, col="post")
        
    def save(self, path:str='', froot:str=None, overwrite=False, config=None):
        fname1 = path + froot + '.csv'
        fname2 = path + froot + '.json'
        #today = date.today().isoformat()
        
        if (os.path.isfile(fname1) or os.path.isfile(fname2)) and not overwrite:
            print("Data not saved. File exists and overwrite=False")
        else:
            self.data.drop(columns=['tokens']).to_csv(fname1)    
            config_data = {
                'global_parameters': self.global_parameters,
                'topic_id': self.topic['id'],
                'config': config
            }
            with open(fname2, 'w') as outfile:
                json.dump(config_data, outfile,indent=4)

    def load_topic(self, fname:str=None, tokenizer: GPT2Tokenizer = None):
        if not os.path.isfile(fname):
            print("Topic-file not found: "+fname)
            return False
        else:
            with open(fname) as f:
                topic = json.load(f)
            self.topic = topic
            if tokenizer != None:
                topic['intro_tokens'] = tokenizer(topic['intro'])['input_ids']
                topic['prompt_tokens'] = tokenizer(topic['prompt'])['input_ids']
                # filter initial posts
                initial_posts = topic['initial_posts']
                initial_posts = [p for p in initial_posts if len(tokenizer(p['text'])['input_ids'])<self.max_tokens_per_initial_claim]
                topic['initial_posts'] = initial_posts
                # tokenize prompt and claims
                topic['claim_tokens'] = {
                    'connector': tokenizer(topic['claims']['connector'])['input_ids'],
                    'pro':[tokenizer(t)['input_ids'] for t in topic['claims']['pro']], # list of token lists
                    'con':[tokenizer(t)['input_ids'] for t in topic['claims']['con']] # list of token lists
                }
            return True
        
        
class LMUtilitiesMixIn:
    """Utilities for language modeling and generation"""
    
    NEWLINE_TOKENID = 198
    EOS_TOKENIDS = [0, 30, 13]
    ETC_TOKENID = 986
    XA0_TOKENID = 1849

    def conditional_loss(self, labels, logits, k_last_tokens):
        # Shift so that logits at index n predict token n in labels
        shift_logits = logits[..., :-1, :].contiguous() # drop last token
        shift_labels = labels[..., 1:].contiguous() # drop first token; this effectively causes the shift

        # Only keep last k_last_tokens tokens
        shift_logits = shift_logits[..., -k_last_tokens:, :].contiguous()
        shift_labels = shift_labels[..., -k_last_tokens:].contiguous() 

        # use CrossEntropy loss function
        loss_fct = torch.nn.CrossEntropyLoss()
        last_k_loss = loss_fct(
            # Flatten the tokens, i.e. reduce dimension
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ) 

        return last_k_loss 


    
    def post_process(self, tokenids):
        """post-process generated post (tokens)"""
        processed = tokenids

        # remove quotation marks 
        processed = [t for t in processed if t not in self.tokenizer("\"")['input_ids']]
        processed = [t if t not in self.tokenizer(".\"")['input_ids'] else self.tokenizer(".")['input_ids'][-1] for t in processed]
        # remove etc.
        processed = [t for t in processed if not t==self.ETC_TOKENID]

        # discard everything after tokenizer.eos_token_id
        if self.tokenizer.eos_token_id in processed:
            eos_id = processed.index(self.tokenizer.eos_token_id)
            processed = processed[:eos_id]

        # post-process sentences
        eos_idx = [i for i,token in enumerate(processed) if token in self.EOS_TOKENIDS]   
        if len(eos_idx)>0:
            # drop last, incomplete sentence
            processed = processed[:(eos_idx[-1]+1)]

            # drop first sentence that repeats n-gram and following ones
            def repeats_ngram(start,stop):
                grams = list(ngrams(processed[start:stop], self.conversation.global_parameters['n_gram_prohibition']))
                grams_before = list(ngrams(processed[:start], self.conversation.global_parameters['n_gram_prohibition']))
                repeats = any(g in grams_before for g in grams)
                return repeats

            drop_later = [start for start,stop in zip(eos_idx[:-1],eos_idx[1:]) if repeats_ngram(start,stop)]
            if len(drop_later)>0:
                processed = processed[:(drop_later[0]+1)]
        else:
            def is_repetitive(l):
                if len(l)==0:
                    return False
                # check whether end of tokenlist l is sufficiently diverse:
                repetitive = any(len(set(l[-i:]))/len(l[-i:])<=.34 for i in range(1,len(l)+1))
                return repetitive
            # index from which on token list is repetitive:    
            j = next((i for i in range(len(processed)) if is_repetitive(processed[:i])), len(processed)+1)
            processed = processed[:j-1] + [self.ETC_TOKENID]

        return processed        
        

        
        
        
        
class AbstractLMAgent:
    """Interface for different LM agents"""
    
    def initialize(self, initial_steps:int=2):
        pass
        
    def update_peers(self, t: int):
        pass

    def update_perspective(self, t: int):
        pass

    def make_contribution(self, t: int):
        pass

    def update_opinion(self, t: int):
        pass


    
    
    
class ListeningLMAgent(AbstractLMAgent,LMUtilitiesMixIn):
    def __init__(self, 
                 model: GPT2LMHeadModel = None, 
                 tokenizer: GPT2Tokenizer = None, 
                 conversation: Conversation = None,
                 neighbors:List[int] = [],
                 agent:int = 0,
                 peer_selection_args:dict = None,
                 perspective_expansion_method:str = 'random',
                 decoding_args:dict = None
                ):
        self.model = model
        self.tokenizer = tokenizer   
        self.conversation = conversation
        self.neighbors = neighbors
        self.agent = agent
        self.peer_selection_args = peer_selection_args 
        self.decoding_args = decoding_args
        self.perspective_expansion_method = perspective_expansion_method
        

        
    # initialize the conversation
    def initialize(self, initial_steps:int=None):
        # initialize peers
        # in the initialization phase, k neighbours are peers
        if initial_steps==None:
            initial_steps = self.conversation.global_parameters.get('n_initial_posts')
        k = round(self.conversation.global_parameters.get('initial_neighb-peer_ratio')*len(self.neighbors))
        initial_peers = random.sample(self.neighbors,k=k)   
        if self.agent not in initial_peers:
            initial_peers.append(self.agent)
        for t in range(initial_steps):
            self.conversation.contribute(
                contribution=initial_peers,
                t=t,
                agent=self.agent,
                col="peers"
            )
        
        # initialize perspective from initial_peers
        for t in range(1,initial_steps):
            post_refs = [(i,j) for i in range(t) for j in initial_peers]
            if len(post_refs)>self.conversation.global_parameters.get('context_size'):
                post_refs = random.sample(post_refs,k=self.conversation.global_parameters.get('context_size'))

            perspective:Perspective = [{'pst':pr,'tst':0} for pr in post_refs]
                    
            self.conversation.contribute(
                contribution=perspective,
                t=t,
                agent=self.agent,
                col="perspective"
            )
            
        
        # initialize posts from topic
        for t in range(initial_steps):
            post = random.choice(self.conversation.topic['initial_posts'])
            self.conversation.submit_post(
                post=post,
                t=t,
                agent=self.agent
            )
            self.conversation.contribute(
                contribution=self.tokenizer(post['text'])['input_ids'],
                t=t,
                agent=self.agent,
                col="tokens"
            )
            
        # initialize opinions
        for t in range(initial_steps):
            self.update_opinion(t)
                    
        
    def update_peers(self, t: int):
        if self.peer_selection_args.get('id')=='all_neighbors':
            self.conversation.contribute(
                contribution=self.neighbors,
                t=t,
                agent=self.agent,
                col="peers"
            )
        elif self.peer_selection_args.get('id')=='bounded_confidence':
            epsilon = self.peer_selection_args.get('epsilon')
            opinion = lambda i: self.conversation.get(agent=i, t=t-1, col='polarity')
            peers = [i for i in self.neighbors if abs(opinion(i)-opinion(self.agent))<epsilon]
            self.conversation.contribute(
                contribution=peers,
                t=t,
                agent=self.agent,
                col="peers"
            )
        else:
            print('Unknown peer selection method! No peers selected.')

            
    def update_perspective(self, t: int):
        # 1. get previous perspective
        perspective_old:Perspective = self.conversation.get(agent=self.agent, t=t-1, col='perspective')
        logging.debug('1. Agent {}: len perspective = {} ({})'.format(self.agent, len(perspective_old), len(set([pp['pst'] for pp in perspective_old]))))


        # 2. forget some former posts
        perspective:Perspective = self.concat_persp(perspective_old, t=t)
        logging.debug('2. Agent {}: len perspective = {} ({})'.format(self.agent, len(perspective), len(set([pp['pst'] for pp in perspective]))))


        # 3. fill-in missing gaps
        perspective:Perspective = self.expand_persp(perspective, t=t)
        logging.debug('3. Agent {}: len perspective = {} ({})'.format(self.agent, len(perspective), len(set([pp['pst'] for pp in perspective]))))


        # 4. remove duplicates
        persp_no_dupl:Perspective = []
        for p in perspective:
            if not p in persp_no_dupl:
                persp_no_dupl.append(p)
        perspective = persp_no_dupl
        logging.debug('4. Agent {}: len perspective = {} ({})'.format(self.agent, len(perspective), len(set([pp['pst'] for pp in perspective]))))


        # 5. update
        self.conversation.contribute(
            contribution=perspective,
            t=t,
            agent=self.agent,
            col="perspective"
        )
        
        
        

    def concat_persp(self, perspective:Perspective, t:int=0) -> Perspective: 
        perspective:Perspective = perspective
        dep_exp = self.conversation.global_parameters.get('relevance_deprecation')
        sc_fact = self.conversation.global_parameters.get('self_confidence')
        m_loss = self.conversation.global_parameters.get('memory_loss')
    
        # weights are used to determine probability that post is retained and not forgotten
        # relevance deprecation and self-confidence
        weight = lambda pp: dep_exp**(pp['tst']) * (sc_fact if pp['pst'][1]==self.agent else 1)
        weights = [weight(pp) for pp in perspective] 

        # rescale weights acc to confirmation bias
        #   rescaling reflects relevance confirmation of current
        #   normalized belief by perspective - post
        mean_weights = sum(weights)/len(weights)
        if self.perspective_expansion_method=='confirmation_bias':
            x0 = self.conversation.get(t=0,agent=self.agent,col="polarity") # baseline belief
            #cb_exp = self.conversation.global_parameters.get('conf_bias_exponent') # exponent
            
            ## elicit opinion batch
            persp_posts = [pp['pst'] for pp in perspective]
            persp_batch = [[p] for p in persp_posts]
            persp_batch = [persp_posts] + persp_batch # add current perspective to batch
            op_batch, _  = self.elicit_opinion_batch(persp_batch)
            #print(op_batch)
            opinion = op_batch[0] # opinion given default perspective
            op_batch = op_batch[1:] # opinions given perspective - indivdual post
            def conf(x):
                c = np.log(x)-np.log(x0) if opinion>x0 else np.log(x0)-np.log(x)
                return c
            weights_conf = [conf(x) for x in op_batch]  
            # add disconf values to weights
            weights = [w1+w2 for w1,w2 in zip(weights, weights_conf)]        
            # finally, mean-rescale:    
            mean_new_weights = sum(weights)/len(weights) 
            weights = [(mean_weights/mean_new_weights)*w for w in weights]

        # sample new perspective according to weights
        new_perspective:Perspective = []
        if m_loss==0:
            for p,w in zip(perspective,weights):
                if random.uniform(0,1)<w:
                    new_perspective.append(p)
        else:
            new_perspective = perspective

        # if no post has been forgotten so far, drop m_loss posts
        if len(perspective)==len(new_perspective) and len(perspective)>0:
            p_drop = random.choices(perspective, k=m_loss, weights=[1-w for w in weights])
            new_perspective = [p for p in perspective if not p in p_drop]                
                
        # increase time-stamp in all posts retained

        new_perspective = [{'pst':pp['pst'],'tst':pp['tst']+1} for pp in new_perspective]

        return new_perspective


    
    def expand_persp(self, perspective:Perspective, t:int=0) -> Perspective:
        perspective:Perspective = perspective
        size = self.conversation.global_parameters.get('context_size')
        
        # add recent contribution of agent herself
        if len(perspective)<size:
            if self.conversation.get(agent=self.agent, t=t-1, col='post') != None:
                perspective = perspective + [{'pst':(t-1,self.agent),'tst':0}]
        
        # list of posts referenced in current perspective
        persp_posts = [pp['pst'] for pp in perspective]

        # peer posts
        peer_posts = self.get_peer_posts(t) # all posts from which new posts that will be added to perspective are chosen
        peer_posts = [p for p in peer_posts if not p in persp_posts] # exclude posts already in perspective

        # DEBUG
        #print('Agent {}: perspective = {} ({}), peer posts = {} ({}).'.format(self.agent, len(perspective), len(set(perspective)), len(peer_posts), len(set(peer_posts))))        

        # determine weights for selecting new posts for perspective according to perspective_expansion_method
        if self.perspective_expansion_method=='random':
            # uniform weights
            weights = [1]*len(peer_posts)

        elif self.perspective_expansion_method=='confirmation_bias':
            # weights reflect relevance confirmation of current normalized belief by post
            x0 = self.conversation.get(t=0,agent=self.agent,col="polarity") # baseline belief
            cb_exp = self.conversation.global_parameters.get('conf_bias_exponent') # exponent
            
            ## elicit opinion batch
            persp_batch = [persp_posts + [pp] for pp in peer_posts]
            persp_batch = [persp_posts] + persp_batch # add contracted perspective to batch
            op_batch, _  = self.elicit_opinion_batch(persp_batch)
            opinion = op_batch[0] # opinion given contracted perspective, no peer post added
            op_batch = op_batch[1:] # opinions given perspective + indivdual peer post

            ## weights depend on confirmation
            def conf(x):
                c = x-x0 if opinion>x0 else x0-x
                c = 0 if c<0 else c
                return c
            weights = [conf(x)**cb_exp for x in op_batch]
            
            # are some weights >0? if not, use uniform positive weights
            if all(w==0 for w in weights):
                weights = [1]*len(peer_posts)

        elif self.perspective_expansion_method=='confirmation_bias_lazy':
            # we select up to k=size-len(persp_posts) peer_posts whose weights will be set to 1: 
            #   a. sample k peer peer posts (sample_a), if each is confirming, then add all, else:
            #   [b. sample another k peer posts (sample_b) and add the k most confirming ones of the 2k posts]   

            x0 = self.conversation.get(t=0,agent=self.agent,col="polarity") # baseline belief
            k = size-len(persp_posts)
            idx_all = list(range(len(peer_posts)))

            if k>=len(peer_posts):
                idx = idx_all
            else:
                sample_a = random.sample(idx_all,k=k)
                ## elicit opinion batch A
                persp_batch = [persp_posts + [peer_posts[i]] for i in sample_a]
                #persp_batch = [persp_posts] + persp_batch # add contracted perspective to batch
                op_batch, _  = self.elicit_opinion_batch(persp_batch)
                #opinion = op_batch[0] # opinion given contracted perspective, no peer post added
                opinion = self.conversation.get(t=t-1,agent=self.agent,col="polarity") # belief at t-1
                #op_batch = op_batch[1:] # opinions given perspective + indivdual peer post
                ## confirmation measure, given conditional opinion x
                def conf(x):
                    c = x-x0 if opinion>x0 else x0-x
                    c = 0 if c<0 else c
                    return c
                conf_a = [conf(x) for x in op_batch]
                #idx = [i for i,x in zip(sample_a,conf_a) if x>0]
                if all(x>0 for x in conf_a):
                    idx=sample_a
                else: 
                    sample_b = random.sample([i for i in idx_all if not i in sample_a],k=min(k,len(peer_posts)-k))
                    ## elicit opinion batch B
                    persp_batch = [persp_posts + [peer_posts[i]] for i in sample_b]
                    op_batch, _  = self.elicit_opinion_batch(persp_batch)
                    conf_b = [conf(x) for x in op_batch]
                    idx = [i for x,i in sorted(zip(conf_a+conf_b,sample_a+sample_b)) if x>0] #sort filtered indices of peer posts by conf value
                    idx = idx[:k]

            weights = [1 if i in idx else 0 for i in idx_all]


        elif self.perspective_expansion_method=='homophily':
            # homophily_exponent
            h_exp = self.conversation.global_parameters.get('homophily_exponent') # exponent
            # all peers
            peers = self.conversation.get(
                t=t,
                agent=self.agent,
                col="peers"
            )
            # exclude agent
            peers = [p for p in peers if p!=self.agent]
            # opinion of agent i
            opinion = lambda i: self.conversation.get(agent=i, t=t-1, col='polarity')
            # similarity of agent i with self.agent
            sim = lambda i: (1-abs(opinion(i)-opinion(self.agent)))
            peer_weights = [sim(p)**h_exp for p in peers]
            peer_weights = [w/sum(peer_weights) for w in peer_weights]
            # choose partner to interact with            
            partner = random.choices(peers, k=1, weights=peer_weights)[0]
            partner_p:Perspective = self.conversation.get(
                    t=t-1,
                    agent=partner,
                    col="perspective"
                )
            partner_p = [pp['pst'] for pp in partner_p]
            # set weight of all partner posts to 1, others to 0
            weights = [1 if p in partner_p else 0 for p in peer_posts]

        else:
            print('Unknown perspective_expansion_method, using uniform weights')
            weights = [1]*len(peer_posts)


        # fill up perspective with peer posts given weights
        ppws = list(set(zip(peer_posts,weights)))
        ppws = [(pp,w) for pp,w in ppws if w>0] # delete zero-weight entries
        if len(ppws)<=(size-len(perspective)):
            ## add all non-zero weight entries
            perspective = perspective + [{'pst':p,'tst':0} for p,_ in ppws]
        else:
            while len(perspective)<size:
                p_new:List[Tuple] = random.choices(ppws, k=1, weights=[w for _,w in ppws]) # draw new post
                perspective = perspective + [{'pst':p_new[0][0],'tst':0}] # add post to perspective
                ppws = [pw for pw in ppws if not pw in p_new] # remove post from ppws (-> sampling without replacement)
                
        return perspective

    

    def get_peer_posts(self, t:int=0):
        """Peer posts at step t
        
        List of all posts in format (step,agent) that are eligible for being newly added to 
        the agent's perspective at step t, here: all posts in the perspectives of peers at step t-1
        """
        
        peers = self.conversation.get(
                t=t,
                agent=self.agent,
                col="peers"
            )

        # we don't exclude agent herself at this stage
        # peers = [i for i in peers if i!=self.agent]        

        peer_posts = []
        for peer in peers:
            # 1. peer_posts include all posts in peers' perspectives
            ppersp:Perspective = self.conversation.get(
                t=t-1,
                agent=peer,
                col="perspective"
            )
            peer_posts = peer_posts + [pp['pst'] for pp in ppersp] 

            # 2. add posts that peers have contributed at previous step
            if self.conversation.get(agent=peer, t=t-1, col='post') != None:
                peer_posts = peer_posts + [(t-1,peer)]

        peer_posts_no_dupl = []
        for pp in peer_posts:       
            if not pp in peer_posts_no_dupl:
                peer_posts_no_dupl.append(pp)

        return peer_posts_no_dupl        
        

        
    def update_opinion(self, t: int):
        perspective:Perspective = self.conversation.get(agent=self.agent, t=t, col='perspective')
        polarity, salience = self.elicit_opinion(perspective)            
        self.conversation.contribute(contribution=polarity, t=t, agent=self.agent, col="polarity")
        self.conversation.contribute(contribution=salience, t=t, agent=self.agent, col="salience")

        
        
    def elicit_opinion(self, perspective: Perspective):
        
        # list of posts referenced in current perspective
        persp_posts: List[Tuple[int]] = [pp['pst'] for pp in perspective]

        # collect_and_glue_perspective_tokens
        token_ids_cond = self.conversation.topic['intro_tokens']
        for tt,i in persp_posts:
            token_ids_cond = token_ids_cond + self.conversation.get(agent=i, t=tt, col='tokens')
        token_ids_cond = token_ids_cond + self.conversation.topic['claim_tokens']['connector']

        # we account for multiple formulations of the pro-claim / the con-claim
        token_ids_pro = self.conversation.topic['claim_tokens']['pro'] # list of token lists
        token_ids_con = self.conversation.topic['claim_tokens']['con'] # list of token lists
        
        # average conditional perplexity pro claims
        PP_pro = 0
        for token_ids in token_ids_pro: 
            input_tensor = torch.tensor([token_ids_cond + token_ids]).to('cuda')
            output = self.model(input_tensor,labels=input_tensor)
            PP_pro = PP_pro + np.exp(self.conditional_loss(input_tensor, output['logits'], len(token_ids)).tolist())
        PP_pro = PP_pro / len(token_ids_pro)

            
        # average conditional perplexity con claims
        PP_con = 0
        for token_ids in token_ids_con: 
            input_tensor = torch.tensor([token_ids_cond + token_ids]).to('cuda')
            output = self.model(input_tensor,labels=input_tensor)
            PP_con = PP_con + np.exp(self.conditional_loss(input_tensor, output['logits'], len(token_ids)).tolist())
        PP_con = PP_con / len(token_ids_con)
        
        polarity = PP_con/(PP_pro+PP_con)
        salience = (PP_pro+PP_con)/2.0
        
        return polarity, salience
            

                
#    def elicit_opinion_batch(self, perspectives:[[(int)]]):
    def elicit_opinion_batch(self, perspectives:List[List[Tuple[int]]]):
        
        batch_size = len(perspectives)
        fwd_batch_size = self.conversation.global_parameters.get('fwd_batch_size',1)
        
        
        token_ids_cond_batch = []
        # collect_and_glue_perspective_tokens
        for persp_posts in perspectives:
            # list of posts referenced in current perspective
            token_ids_cond = self.conversation.topic['intro_tokens']
            for tt,i in persp_posts:
                token_ids_cond = token_ids_cond + self.conversation.get(agent=i, t=tt, col='tokens')
            token_ids_cond = token_ids_cond + self.conversation.topic['claim_tokens']['connector']
            token_ids_cond_batch.append(token_ids_cond)


        # we account for multiple formulations of the pro-claim / the con-claim
        token_ids_pro = self.conversation.topic['claim_tokens']['pro'] # list of token lists
        token_ids_con = self.conversation.topic['claim_tokens']['con'] # list of token lists
        

        # average conditional perplexity of every `token_ids_claim` given condition in batch `token_ids_cond_batch`
        def aver_cond_pp_batch(token_ids_claim):            
            PP = [0]*batch_size
            for token_ids in token_ids_claim: 
                input_tokens_batch = [token_ids_cond + token_ids for token_ids_cond in token_ids_cond_batch]
                logging.debug('Size of input_tokens_batch: %s'%(len(input_tokens_batch)))
                mini_batches = [input_tokens_batch[i:i+fwd_batch_size] for i in range(0, len(input_tokens_batch), fwd_batch_size)]
                logging.debug('Number of mini_batches: %s'%(len(mini_batches)))
                PP_claim = []
                for mini_batch in mini_batches:
                    logging.debug('Size of mini_batch: %s'%(len(mini_batch)))
                    max_len = max([len(x) for x in mini_batch])
                    # pad mini batch:
                    mini_batch_padded = [([self.tokenizer.pad_token_id]*(max_len-len(x))) + x for x in mini_batch]
                    if len(set(len(e) for e in mini_batch_padded))!=1:
                        logging.debug("Error! Size of token lists in minibatch: "+str([len(e) for e in mini_batch_padded]))
                    input_tensor = torch.tensor(mini_batch_padded).to('cuda')
                    output = self.model(input_tensor,labels=input_tensor)
                    PP_mb = []
                    for label,logits in zip(torch.split(input_tensor,1), torch.split(output['logits'],1)):
                        PP_mb.append(
                            np.exp(self.conditional_loss(label, logits, len(token_ids)).tolist())
                        )
                    PP_claim = PP_claim + PP_mb
                PP = [x+y for x,y in zip(PP, PP_claim)]
            PP = [x/len(token_ids_claim) for x in PP]
            return PP
                                
        # average conditional perplexity pro claims
        PP_pro = aver_cond_pp_batch(token_ids_pro)
            
        # average conditional perplexity con claims
        PP_con = aver_cond_pp_batch(token_ids_con)
        
        polarity_batch = [y/(x+y) for x,y in zip(PP_pro,PP_con)]
        salience_batch = [(x+y)/2 for x,y in zip(PP_pro,PP_con)]
        
        return polarity_batch, salience_batch
            
    


    
class GeneratingLMAgent(ListeningLMAgent):
        
    # currently not used:
    def get_peer_posts2(self, t:int=0):
        """Peer posts at step t
        
        List of all posts in format (step,agent) that are eligible for being newly added to 
        the agent's perspective at step t, here: all new contributions by peers at step t-1
        """
        
        peers = self.conversation.get(
                t=t,
                agent=self.agent,
                col="peers"
            )
        peer_posts = [(t-1,i) for i in peers]
        return peer_posts
    
    
    
    def make_contribution(self, t: int):
        perspective:Perspective = self.conversation.get(agent=self.agent, t=t, col='perspective')        
        
        # collect_and_glue_perspective_tokens
        tokens = self.conversation.topic['intro_tokens']
        tokens = tokens + [self.NEWLINE_TOKENID]
        for tt,i in [pp['pst'] for pp in perspective]:
            tokens = tokens + self.conversation.get(agent=i, t=tt, col='tokens')
            tokens = tokens + [self.NEWLINE_TOKENID]
        tokens = tokens + self.conversation.topic['prompt_tokens']
        
        # adjust max length parameter
        params = self.decoding_args.copy()
        params['max_length'] = params['max_length']+len(tokens)
        params['pad_token_id'] = self.tokenizer.pad_token_id
        
        # generate
        output = self.model.generate(
            torch.tensor([tokens]).to('cuda'),
            **params
        )
        
        gen_tokens = output.tolist()[0][len(tokens):] # drop input sequence
        gen_tokens = self.post_process(gen_tokens) # post-process
        
        # decode
        gen_text = {'text':self.tokenizer.decode(gen_tokens)}

        
        self.conversation.submit_post(post=gen_text, t=t, agent=self.agent)
        self.conversation.contribute(contribution=gen_tokens, t=t, agent=self.agent, col="tokens")        
        
        

    
class FormalModelAgent(ListeningLMAgent):
    """modeling posts and opinions purely formally"""

    def elicit_opinion(self, perspective: Perspective):
        reason_strengths:dict = self.conversation.reason_strengths
        if reason_strengths==None:
            print("Error: reason_strengths of conversation not initialized!")
        pp_rs = [reason_strengths.get(p['pst']) for p in perspective]
        return np.mean(pp_rs), 1

    def elicit_opinion_batch(self, perspectives:List[Tuple[int]]):
        opinions = [self.elicit_opinion([{'pst':post,'tst':0} for post in pp]) for pp in perspectives]
        polarity_batch = [x for x,_ in opinions]
        salience_batch = [y for _,y in opinions]
        return polarity_batch, salience_batch


