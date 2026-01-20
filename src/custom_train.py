# === import libraries ===
import random
import json
from pathlib import Path
import sys
from tqdm import tqdm #to track process
import spacy
from spacy.training.initialize import init_nlp
from spacy.util import load_config, minibatch
from spacy.training.example import Example
from spacy.training import load_corpus
from spacy.scorer import Scorer

#python custom_train.py train_fold0.spacy dev_fold0.spacy output_fold0
train_path = Path(sys.argv[1])
dev_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

#load congfic.cfg file
config_path = Path("/content/ner_drugs/configs/config.cfg")
config = load_config(config_path)

#create, check and load checkpoints
output_dir.mkdir(parents=True, exist_ok=True) #create checkpoint folder
checkpoint_dir = output_dir
latest_checkpoint = None
start_iter = 0
n_iter = 10  #total training iterations

if checkpoint_dir.exists():
    #checkpoints sorted in descending order 
    checkpoints = sorted(
        checkpoint_dir.glob("model_checkpoint_iter_*"),
        key=lambda p: int(p.name.split("_")[-1]), #checks iter_number
        reverse=True #desc
    )
    #load latest checkpoint that's been sorted to checkpoints[0]
    if checkpoints:
        latest_checkpoint = checkpoints[0]
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        nlp = spacy.load(str(latest_checkpoint)) #loads last saved model
        start_iter = int(latest_checkpoint.name.split("_")[-1])
    else:
        nlp = init_nlp(config, use_gpu=True) #else initialize model afresh
else:
    nlp = init_nlp(config, use_gpu=True) #else initialize model afresh

#load training corpus
train_corpus = load_corpus(config["training"]["train_corpus"], config)
dev_corpus = load_corpus(config["training"]["dev_corpus"], config)

train_samples = list(train_corpus(nlp))
dev_samples = list(dev_corpus(nlp))

#logic to resume training from last iteration
if start_iter == 0:
    print(f">>> Initializing with {len(train_samples)} training examples")
    optimizer = nlp.initialize(lambda: train_samples) #initialize pipeline & return optimizer to update the model weights
    print(">>> Optimizer initialized:", type(optimizer))
else:
    optimizer = nlp.resume_training() 
    print(">>> Optimizer resumed.")

#train and evaluate loop with checkpointing
batch_sizes = config["training.batcher"]["size"]
eval_every = 2  #create checkpoint after every 2 iterations

for i in tqdm(range(start_iter, n_iter), desc="Training Progress"):
    losses = {}
    random.shuffle(train_samples)
    batches = minibatch(train_samples, size=batch_sizes)

    #update nlp with current model parameter
    for batch in batches:
        nlp.update(batch, sgd=optimizer, losses=losses, drop=0.25) 
        #optimizer holds state between updates
        #drop = dropout rate to randomly drop indiv features and representations

    print(f"\n>>> Iteration {i+1}, Losses: {losses}")

    #evaluate
    scorer = Scorer()
    examples = []
    for example in dev_samples:
        pred = nlp(example.text) #run nlp pipeline on each row's raw text in dev_samples set
        example_pred = Example(pred, example.reference) #align nlp.pred results with actual results
        examples.append(example_pred) #add to list
    scores = scorer.score(examples) #evaluate all in list
    print(f">>> Evaluation scores at Iteration {i+1}: {scores}")

    #save checkpoints and eval metrics at that checkpoint
    if (i + 1) % eval_every == 0 or (i + 1) == n_iter:
        checkpoint_path = checkpoint_dir / f"model_checkpoint_iter_{i+1}" 
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(str(checkpoint_path))
        print(f"Checkpoint saved")

        eval_path = checkpoint_dir / f"metrics_iter_{i+1}.json"
        with open(eval_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Evaluation metrics saved") 

# === FINAL MODEL SAVE ===
final_model_path = checkpoint_dir / "final_model"
final_model_path.mkdir(parents=True, exist_ok=True)
nlp.to_disk(final_model_path)
print(f"Final model saved to: {final_model_path.resolve()}")

