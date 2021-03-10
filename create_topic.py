# python script for creating topics from Kialo debates

# %%

import json
import re

kialo_path = '/Users/ggbetz/Documents/Philosophie/vortraege_paper/2019/Artificial Epistemic Agent/kialo_corpus/raw debates/'

topic = {
        'id': 'compulsory_voting',
        'claims': {
            'connector': ' And therefore, all in all,',
            'pro': [' compulsory voting is a pretty good idea, and I am for compulsory voting.'],
            'con': [' compulsory voting is a really bad idea, and I am against compulsory voting.']
        },
        'intro': 'Let\'s discuss compulsory voting! Should citizens be free to abstain from voting? Or should voting be made compulsory?',
        'prompt': 'I more or less agree with what my peers are saying here. Regarding the idea of compulsory voting, I\'d just add the following thought:\n',
        'initial_posts': [
            { 
                'type':'pro',
                'text':'Turnout in British election is distressingly low. Compulsory voting might fix this problem and increase the legitimacy of our government.'
            }            
        ]
    }
# %%



templates = [
    {
        'filename': 'a-society-with-no-gender-would-be-better-16617.txt',
        'id': 'society-with-no-gender',
        'pro': [' a society without gender is a pretty good idea, there is no need to distinguish between men and women.'],
        'con': [' a society without gender is a really bad idea, there are important differences between men and women we should pay attention to.'],
        'intro': 'Let\'s discuss a society without gender!',
        'prompt': 'I more or less agree with what my peers are saying here. Regarding the idea of a society without gender, I\'d just add the following thought:\n',
    },
    {
        'filename': 'all-drugs-should-be-legalized-7100.txt',
        'id': 'all-drugs-should-be-legalized',
        'pro': [' legalization of drugs is a pretty good idea, all drugs should be legal.'],
        'con': [' legalization of drugs is a really bad idea, drugs should remain illegal.'],
        'intro': 'Let\'s discuss legalization of drugs!',
        'prompt': 'I more or less agree with what my peers are saying here. Regarding the legalization of drugs, I\'d just add the following thought:\n',
    }    
]

for template in templates:

    new_topic = topic.copy()
    new_topic['id'] = template.get('id')
    new_topic['claims']['pro'] = template.get('pro')
    new_topic['claims']['con'] = template.get('con')
    new_topic['intro'] = template.get('intro')
    new_topic['prompt'] = template.get('prompt')
    new_topic['initial_posts'] = []


    with open(kialo_path+template.get('filename')) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    reason_pattern = r"1.(?:[\d|.]* (?:Pro: |Con: ))"

    for line in content:
        rm = re.search(reason_pattern, line)
        if rm!=None:
            if rm.start()==0:
                r = line[rm.end():]
                if not r.startswith('->'):
                    r = r.replace('\(','(')
                    r = r.replace('\)',')')
                    new_topic['initial_posts'].append({'text':r})

    with open('topics/'+template.get('filename')[:-4]+'.json', 'w') as outfile:
        json.dump(new_topic, outfile,indent=4)

# %%
len(new_topic['initial_posts'])
# %%
