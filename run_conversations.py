# python script file for running and storing a conversation

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import logging

import os
import os.path
import argparse

from nltk import ngrams

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from beldynlm import *



parser = argparse.ArgumentParser()
parser.add_argument("--model", help="hugging face model'")
parser.add_argument("--ensemble_id", help="also used as results folder'")
parser.add_argument("--run_id", help="results file name")
parser.add_argument("--agent_type", help="'listening' or 'generating' or 'formal'")
parser.add_argument("--decoding_key", help="key of decoding profile, e.g. 'narrow_1'")
parser.add_argument("--topic", help="topic file (json)")
parser.add_argument("--n_agents", help="number of agents", type=int)
parser.add_argument("--max_t", help="number of time steps to simulate", type=int)
parser.add_argument("--n_initial_posts", help="number of initial steps", type=int)
parser.add_argument("--background_beliefs", help="number of initial posts that will be retained as immutable background beliefs in the context, < context_size", type=int)
parser.add_argument("--context_size", help="size of perspective (max posts)", type=int)
parser.add_argument("--memory_loss", help="max number of posts forgotten per interaction, 0: endogeneously determined", type=int)
parser.add_argument("--self_confidence", help="factor that controls forgetting", type=int)
parser.add_argument("--perspective_expansion_method", help="'random', 'homophily' or 'confirmation_bias'")
parser.add_argument("--peer_selection_method", help="'all_neighbors' or 'bounded_confidence'")
parser.add_argument("--conf_bias_exponent", help="conf_bias_exponent, used only with confirmation_bias", type=float)
parser.add_argument("--homophily_exponent", help="homophily_exponent, used only with homophily", type=float)
parser.add_argument("--epsilon", help="epsilon, used only with bounded_confidence", type=float)
parser.add_argument("--relevance_deprecation", help="relevance_deprecation", type=float)
args = parser.parse_args()

ENSEMBLE_ID = args.ensemble_id
RUN_ID = args.run_id
RESULT_DIR = 'results/'+ENSEMBLE_ID
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    
    
global_parameters = {
    'model':args.model if args.model else 'distilgpt2',
    'topic':args.topic if args.topic else 'topics/compulsory_voting_polarized.json',
    'agent_type':args.agent_type if args.agent_type else 'listening',
    'n_agents':args.n_agents if args.n_agents else 15,
    'max_t':args.max_t if args.max_t else 200,
    'n_initial_posts':args.n_initial_posts if args.n_initial_posts else 5, 
    'initial_neighb-peer_ratio':.5, 
    'background_beliefs':args.background_beliefs if args.background_beliefs else 0, 
    'context_size':args.context_size if args.context_size else 8, 
    'memory_loss':args.memory_loss if args.memory_loss else 0, 
    'relevance_deprecation':args.relevance_deprecation if args.relevance_deprecation else .95,
    'self_confidence':args.self_confidence if args.self_confidence else 1,  
    'contribution_probability':.2, # probability that agent will contribute in given round
    'n_gram_prohibition':5,  
    'perspective_expansion_method':args.perspective_expansion_method if args.perspective_expansion_method else 'random', 
    'conf_bias_exponent':args.conf_bias_exponent if args.conf_bias_exponent else 50,
    'homophily_exponent':args.homophily_exponent if args.homophily_exponent else 50,
    'peer_selection_method':args.peer_selection_method if args.peer_selection_method else 'all_neighbors', 
    'fwd_batch_size':1
}

peer_selection_parameters = [
     {
         'id':'all_neighbors'
     },
     {
         'id':'bounded_confidence',
         'epsilon': args.epsilon if args.epsilon else 0.01
     }
]

decoding_profiles = {
    "narrow_1":{
        'do_sample':True, 
        'num_beams':5,
        'temperature': 1.0,
        'top_p': 0.5, 
        'top_k':0,
        'repetition_penalty':1.2,
        'max_length':40,
        'bad_words_ids':[[LMUtilitiesMixIn.NEWLINE_TOKENID],[LMUtilitiesMixIn.ETC_TOKENID]]
    },
    "creative_1":{
        'do_sample':True, 
        'num_beams':5,
        'temperature': 1.4,
        'top_p': 0.95, 
        'top_k':0,
        'repetition_penalty':1.2,
        'max_length':40,
        'bad_words_ids':[[LMUtilitiesMixIn.NEWLINE_TOKENID],[LMUtilitiesMixIn.ETC_TOKENID]]
    }    
}

decoding_key = args.decoding_key if args.decoding_key else 'creative_1'
if not decoding_key in decoding_profiles:
    decoding_key = 'creative_1'
decoding_parameters = decoding_profiles[decoding_key]



###################
#       LM        #
###################

tokenizer = GPT2Tokenizer.from_pretrained(global_parameters.get('model'))
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(global_parameters.get('model'))
model.to("cuda")



###################
#    INITIALIZE   #
###################

conversation = Conversation(global_parameters=global_parameters)
conversation.load_topic(global_parameters.get('topic'), tokenizer=tokenizer)

LMAgent = GeneratingLMAgent

agents = []
for i in range(global_parameters['n_agents']):
    agent = LMAgent(
        model=model, 
        tokenizer=tokenizer, 
        conversation=conversation, 
        neighbors=list(range(global_parameters['n_agents'])),
        peer_selection_args=next(
            args for args in peer_selection_parameters 
            if args['id']==global_parameters['peer_selection_method']
        ),
        perspective_expansion_method=global_parameters['perspective_expansion_method'],
        decoding_args=decoding_parameters,
        agent=i
    )
    agents.append(agent)

for agent in agents:
    agent.initialize()
    
    
    

###################
#      LOOP       #
###################

for t in tqdm(range(global_parameters['n_initial_posts'],global_parameters['max_t'])):
    for agent in agents:
        # Determine peers
        agent.update_peers(t)

        # Determine perspective
        agent.update_perspective(t)

        # Generate posts
        if global_parameters.get('agent_type')=='generating':
            if random.uniform(0,1)<global_parameters.get('contribution_probability'):
                agent.make_contribution(t)
        
        # Update opinion
        agent.update_opinion(t)



###################
#      SAVE       #
###################

RUN_ID = args.run_id
RESULT_DIR = 'results/'+ENSEMBLE_ID+'/'

conversation.save(
    path=RESULT_DIR, 
    froot=ENSEMBLE_ID+'_'+RUN_ID, 
    overwrite=True, 
    config={'decoding_parameters':decoding_parameters,'peer_selection_parameters':peer_selection_parameters}
)



