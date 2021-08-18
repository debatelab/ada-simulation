# Belief Dynamics with LMs

Multi-agent simulation model of belief dynamics with LMs.

We recommend to install the required packages (`requirements.txt`) in a virtual environment. 

You can run a simulation like, e.g.:

```bash
python run_conversations.py \
  --model "gpt2" \
  --ensemble_id "ensemble1" \
  --run_id "run1" \
  --agent_type "generating" \
  --decoding_key "narrow_1" \
  --topic "topics/all-drugs-should-be-legalized-7100.json" \
  --max_t 150 \
  --n_initial_posts 5 \
  --n_agents 10 \
  --context_size 8 \
  --memory_loss 2 \
  --peer_selection_method "all_neighbors" \
  --perspective_expansion_method "random" \
  --epsilon 1 \
  --conf_bias_exponent 1 \
  --relevance_deprecation .9 
```


We've been using the following bash script to start multiple simulation runs on a HPC cluster (SLURM):

```bash
#!/bin/bash
# run baseline ensemble

# We assume running this from the script directory
job_directory=.

n_agents_=(20 30)
context_size_=(8 12)
memory_loss=2
peer_selection_method="all_neighbors" # "bounded_confidence"
perspective_expansion_method="random" # "confirmation_bias"
epsilon_=(1)  # (.005 .01 .02 .03 .04)
duplicate_runs=50 # number of runs with same parameters
conf_bias_exponent_=(1)  #(5 50)
model="gpt2-xl"
ensemble_id="20210325-02_base"
count=0
agent_type="generating"
decoding_parameters_=("narrow_1") # ("narrow_1" "creative_1")
topic="topics/all-drugs-should-be-legalized-7100.json"
max_t=150
n_initial_posts=5
relevance_deprecation=.9

for n_agents in ${n_agents_[@]}; do

  for context_size in ${context_size_[@]}; do
      
    for decoding_parameters in ${decoding_parameters_[@]}; do

  	      ((count=count+1))
	      job_file="${job_directory}/job_${ensemble_id}-${count}.job"

		  echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=${ensemble_id}-${count}
#SBATCH --partition=gpu_4

count_2=0

module load devel/cuda/10.2

source env/bin/activate

echo ${ensemble_id} ${context_size} ${n_agents} 

for epsilon in ${epsilon_[@]}; do
  
  for conf_bias_exponent in ${conf_bias_exponent_[@]}; do
  
	for (( i=1; i<=${duplicate_runs}; i++ )); do

	  ((count_2=count_2+1))

	  python run_conversations.py --model ${model} --ensemble_id ${ensemble_id} --run_id ${count}-\${count_2} --agent_type ${agent_type} --decoding_key ${decoding_parameters} --topic ${topic} --max_t ${max_t} --n_initial_posts ${n_initial_posts} --n_agents ${n_agents} --context_size ${context_size} --memory_loss ${memory_loss} --peer_selection_method ${peer_selection_method} --perspective_expansion_method ${perspective_expansion_method} --epsilon \${epsilon} --conf_bias_exponent \${conf_bias_exponent} --relevance_deprecation ${relevance_deprecation} 

	done
		
  done
  
done

deactivate" > $job_file
	        sbatch $job_file
	        
    done

  done

done
```

