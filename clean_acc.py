import json
import time
import argparse
import ast
import re
from utils import load_model

# python clean_acc.py

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )

    # Model
    parser.add_argument(
        '--file',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
        default="results/results.json"
    )

    return parser.parse_args()


# start evaluation for clean acc
args = parse_arguments()

with open(args.file, 'r') as f:
    data = json.load(f)

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

score = 0
num = 0
for i in range(len(data)):
    instruction = data[i]["instruction"]
    instruction_split = instruction.split("\n\n")
    instruction = instruction_split[0]+"\n"+instruction_split[1]
    ground_truth_label = data[i]["ground_truth_label"]
    output_label = data[i]["output_vallina"]
    system_prompt = "You are a helpful assistant."
    template = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]".format(question=instruction, answer=output_label)

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": template}]

    chat = load_model(model="gpt4")

    output = None
    s = 0
    while output is None and s<5:
        s+=1
        output = chat.run(messages)
        time.sleep(2)

    # output = chat.run(messages)
    print(f"--------------------------------------{i}: TEMPLATE-------------------------------------------")
    print(template)
    print("\n\n")
    print(f"--------------------------------------{i}: OUTPUT-------------------------------------------")
    print(output)
    print("\n\n")

    match = re.search(one_score_pattern, output)
    if not match:
        match = re.search(one_score_pattern_backup, output)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1

    if rating!=-1:
        score+=rating
        num+=1
    
final_score = score/num

print("total test number: "+str(num))
print("LLM Judge Score: "+str(final_score))