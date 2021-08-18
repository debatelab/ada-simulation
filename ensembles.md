# List of Ensembles

id | size | generation | peer selection | perspective update | etc.
--- | --- | --- | --- | --- | ---
*20210323-01_base* | 157 | no | all | random | gpt2-xl
*20210323-01_bc*   | 163 | no | bounded confidence | random | gpt2-xl
*20210323-02_bc*   | 320 | no | bounded confidence | random | runs like in *20210323-01_bc* with eps>.02 
*20210323-01_cb*   | 152 | no | all | confirmation bias | gpt2-xl
*20210323-01_hp* | 164 | no | all | homophily | gpt2-xl
*20210323-02_hp* | 198 | no | all | homophily | runs like in *20210323-01_hp* with hp_exp=50
*20210325-01_base* | ? | yes | all | random | gpt2-xl
*20210325-01_bc* | ? | yes | bounded confidence | random | gpt2-xl
*20210325-01_cb* | ? | yes | all | confirmation bias | gpt2-xl
*20210325-01_hp* | ? | yes | all | homophily | gpt2-xl
*20210325-02_base* | ? | yes | all | random | gpt2-xl, like *01 with narrow gen.
*20210325-02_bc* | ? | yes | bounded confidence | random | gpt2-xl, like *01 with narrow gen.
*20210325-02_cb* | ? | yes | all | confirmation bias | gpt2-xl, like *01 with narrow gen.
*20210325-02_hp* | ? | yes | all | homophily | gpt2-xl, like *01 with narrow gen.
*20210409-01* | 12*150 | y/n | div | div | gpt2-xl, 3x4 scenarios from april draft
