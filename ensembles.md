# List of Ensembles

id | size | generation | peer selection | perspective update | etc.
--- | --- | --- | --- | --- | ---
*20210323-01_base* | 157? | no | all | random | gpt2-xl
*20210323-01_bc*   | 163? | no | bounded confidence | random | gpt2-xl
*20210323-02_bc*   | x | no | bounded confidence | random | runs like in *20210323-01_bc* with eps>.02 
*20210323-01_cb*   | 123? | no | all | confirmation bias | gpt2-xl
*20210323-01_hp* | 164? | no | all | homophily | gpt2-xl
*20210323-02_hp* | x | no | all | homophily | runs like in *20210323-01_hp* with hp_exp=50
