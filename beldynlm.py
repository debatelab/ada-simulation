import pandas as pd
import numpy as np
import random
import json

import os
import os.path

from nltk import ngrams

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel




class Conversation:
    """Takes care of the data"""
        
    def __init__(self, global_parameters:dict=None, topic:dict=None):
        self.topic = topic
        self.global_parameters = global_parameters
        
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
        

    def contribute(self, contribution=None, agent:int=0, t:int=0, col:str=None):
        self.data.loc[t,agent][col] = contribution

    def get(self, agent:int=0, t:int=0, col:str=None):
        return self.data.loc[t,agent][col]

    def submit_post(self, post=None, agent:int=0, t:int=0):
        self.contribute(contribution=post, agent=agent, t=t, col="post")
        
    def save_conversation(self, path:str=None, froot:str=None, overwrite=False):
        fname1 = path + froot + '.csv'
        fname2 = path + froot + '.json'
        #today = date.today().isoformat()
        
        if (os.path.isfile(fname1) or os.path.isfile(fname2)) and not overwrite:
            print("Data not saved. File exists and overwrite=False")
        else:
            conversation.data.to_csv(fname1)    
            config_data = {
                global_parameters: global_parameters,
                peer_selection_parameters: peer_selection_parameters,
                decoding_parameters: decoding_parameters
            }
            with open(fname2, 'w') as outfile:
                json.dump(config_data, outfile,indent=4)

    def load_topic(self, fname:str=None):
        if not os.path.isfile(fname):
            print("Topic-file not found: "+fname)
            return False
        else:
            with open(fname) as f:
                topic = json.load(f)
            self.topic = topic
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
            processed = processed + [self.ETC_TOKENID]

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
                 neighbors:[int] = [],
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
            perspective = [(i,j) for i in range(t) for j in initial_peers]
            if len(perspective)>self.conversation.global_parameters.get('context_size'):
                perspective = random.sample(perspective,k=self.conversation.global_parameters.get('context_size'))
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
        perspective = self.conversation.get(agent=self.agent, t=t-1, col='perspective')

        # 2. forget some former posts
        perspective = self.concat_persp(perspective, t=t)
        
        # 3. fill-in missing gaps
        perspective = self.expand_persp(perspective, t=t)
        
        # 4. remove duplicates
        perspective = list(set(perspective))
        
        # 5. update
        self.conversation.contribute(
            contribution=perspective,
            t=t,
            agent=self.agent,
            col="perspective"
        )
        

    def concat_persp(self, perspective, t:int=0):
        perspective = perspective
        dep_exp = self.conversation.global_parameters.get('relevance_deprecation')
        sc_fact = self.conversation.global_parameters.get('self_confidence')
        
        weight = lambda tt,i: dep_exp**(t-tt-1) * (sc_fact if i==self.agent else 1)
        weights = [weight(tt,i) for tt,i in perspective] 

        new_perspective = []
        for p,w in zip(perspective,weights):
            if random.uniform(0,1)<w:
                new_perspective.append(p)

        return new_perspective


    
    def expand_persp(self, perspective, t:int=0):
        perspective = perspective
        size = self.conversation.global_parameters.get('context_size')
        peer_posts = self.get_peer_posts(t) # all posts from which new posts that will be added to perspective are chosen

        # append all peer posts if max perspective size allows 
        if len(peer_posts)+len(perspective) <= size:
            return peer_posts + perspective

        # determine weights for selecting new posts for perspective according to perspective_expansion_method
        if self.perspective_expansion_method=='random':
            # uniform weights
            weights = [1]*len(peer_posts)
        elif self.perspective_expansion_method=='confirmation_bias':
            # weights reflect relevance confirmation of current normalized belief by post
            x0 = self.conversation.get(t=0,agent=self.agent,col="polarity") # baseline belief
            cb_exp = self.conversation.global_parameters.get('conf_bias_exponent') # exponent
            
            ## elicit opinion batch
            persp_batch = [perspective + [pp] for pp in peer_posts]
            persp_batch = [perspective] + persp_batch # add contracted perspective to batch
            op_batch, _  = self.elicit_opinion_batch(persp_batch)
            opinion = op_batch[0] # opinion given contracted perspective, no peer post added
            op_batch = op_batch[1:] # opinions given perspective + indivdual peer post
            def conf(x):
                c = (x-x0)/(opinion-x0)
                c = 0 if c<0 else c
                return c
            weights = [conf(x)**cb_exp for x in op_batch]
            
            #opinion_perspective = self.elicit_opinion(perspective)[0] # opinion given contracted perspective, no peer post added
            #def conf(p:(int)):
            #    opinion = lambda persp: self.elicit_opinion(persp)[0] # get polarity
            #    c = (opinion(perspective+[p])-x0)/(opinion_perspective-x0)
            #    c = 0 if c<0 else c
            #    return c                
            #weights = [(conf(p)**cb_exp) for p in peer_posts]            
            
            # are some weights >0? if not, use uniform positive weights
            if all(w==0 for w in weights):
                weights = [1]*len(peer_posts)
        else:
            print('Unknown perspective_expansion_method, using uniform weights')
            weights = [1]*len(peer_posts)

        peer_posts = random.choices(peer_posts, k=(size-len(perspective)), weights=weights)
                
        return perspective + peer_posts

    

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
        peer_posts = []
        for peer in peers:
            ppersp = self.conversation.get(
                t=t-1,
                agent=peer,
                col="perspective"
            )
            peer_posts = peer_posts + ppersp        
        peer_posts = list(set(peer_posts))
        return peer_posts        
        

        
    def update_opinion(self, t: int):
        perspective = self.conversation.get(agent=self.agent, t=t, col='perspective')
        polarity, salience = self.elicit_opinion(perspective)            
        self.conversation.contribute(contribution=polarity, t=t, agent=self.agent, col="polarity")
        self.conversation.contribute(contribution=salience, t=t, agent=self.agent, col="salience")

        
        
    def elicit_opinion(self, perspective: [(int)]):
        
        # collect_and_glue_perspective_tokens
        token_ids_cond = self.conversation.topic['intro_tokens']
        for tt,i in perspective:
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
            

                
    def elicit_opinion_batch(self, perspectives: [[(int)]]):
        
        batch_size = len(perspectives)
        fwd_batch_size = self.conversation.global_parameters.get('fwd_batch_size')
        
        
        token_ids_cond_batch = []
        # collect_and_glue_perspective_tokens
        for perspective in perspectives:
            token_ids_cond = self.conversation.topic['intro_tokens']
            for tt,i in perspective:
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
                mini_batches = [input_tokens_batch[i:i+fwd_batch_size] for i in range(0, len(input_tokens_batch), fwd_batch_size)]
                PP_claim = []
                for mini_batch in mini_batches:
                    max_len = max([len(x) for x in mini_batch])
                    # pad mini batch:
                    mini_batch_padded = [([self.tokenizer.pad_token_id]*(max_len-len(x))) + x for x in mini_batch]
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
        

    def get_peer_posts(self, t:int=0):
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
        perspective = self.conversation.get(agent=self.agent, t=t, col='perspective')        
        
        # collect_and_glue_perspective_tokens
        tokens = self.conversation.topic['intro_tokens']
        tokens = tokens + [self.NEWLINE_TOKENID]
        for tt,i in perspective:
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
        
        

    
    
        