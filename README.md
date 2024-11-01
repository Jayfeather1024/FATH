# FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks

Official Code Repository of the paper: FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks.

## Requirements

- Create our new conda environment:
    ```bash
    conda create -n fath python=3.9
    conda activate fath
    ```

- Installation of required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Set Azure OpenAI API

Set your Azure OpenAI API settings in ``/model_config/gpt.yaml``, including the deployname, endpoint and api_key.

## Run Scripts

Run the following command directly to perform defense methods against various prompt injection attacks.
```
python run_FATH.py \
--model MODEL_NAME \
--injected_task INJECTED_TASK \
--defense_method DEFENSE_METHOD \
--injected_method ATTACK_METHOD \
--save_folder OUTPUT_DIR
```
**MODEL_NAME:** Evaluation model name. "gpt35turbo" for GPT3.5; "llama3" for Llama3.

**INJECTED_TASK:** Injection task name. "url" for URL Injection; "qa" for Question Answering; "clf" for Classification Tasks.

**DEFENSE_METHOD:** Defense method name. "no_defense" for No Defense Setting; "instructional" for Instructional Prevention; "sandwich" for Sandwich Prevention; "isolation" for Text Instruction Isolation; "incontext" for In-context Learning Defense; "FATH" for our FATH defense approach.

**ATTACK_METHOD:** Attack method name. "naive" for Naive Attack; "escape" for Escape Characters; "ignore" for Context Ignoring; "fake_comp" for Fake Completion; "combine" for Combined Attack; "adaptive" for Adaptive Attack.

**OUTPUT_DIR:** results folder name.

## Clean Performance

Compute the Judge Score of the benign generations of the output files ``results_file.json`` generated by ``run_FATH.py`` with the following command:
```
python clean_acc.py --file results_file.json
```

