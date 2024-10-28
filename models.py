import yaml
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AzureOpenAI
import openai
import os
import yaml


class OpensourceLLM:
    def __init__(self, model=""):
        self.model = model
        config_path = os.path.join("model_config/opensource.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config
        self.path = self.config[self.model]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def run(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        output = self.tokenizer.decode(response, skip_special_tokens=True)

        return output

    def run_one_message(self, messages):
        prompt = ""
        for i in range(len(messages)):
            prompt += messages[i]['role'] + ":\n"
            prompt += messages[i]['content'] + "\n"

        output = self.run(messages)

        return prompt, output


class GPT:
    def __init__(self, model=""):
        self.model = model
        config_path = os.path.join("model_config/gpt.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config

    def run(self, messages, max_requests=5):
        temperature = self.config["gpt"]["temperature"]

        self.client = AzureOpenAI(
            api_version = "2023-05-15",
            azure_endpoint = self.config["gpt"][self.model][1]["endpoint"], 
            api_key=self.config["gpt"][self.model][2]["api_key"],
        )

        ans=None
        num_requests = 0
        while ans==None and num_requests<max_requests:
            num_requests+=1
            try:
                ans = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature = temperature
                )
            except openai.BadRequestError:
                ans = None

        if ans != None:
            output = ans.choices[0].message.content
        else:
            output = "None"

        return output

    def run_one_message(self, messages):
        # template = self.config["gpt"]["template"]
        # template = template.format(system=messages[0]["content"], user=messages[1]["content"])
        prompt = ""
        for i in range(len(messages)):
            prompt += messages[i]['role'] + ":\n"
            prompt += messages[i]['content'] + "\n"

        output = self.run(messages, max_requests=5)

        return prompt, output
