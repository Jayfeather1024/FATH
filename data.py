import os
from utils import data_preprocess
import dataset.OpenPromptInjection as PI
from dataset.OpenPromptInjection.utils import open_config
import yaml

class Data:
    def __init__(self, baseline="openprompt", target_task="", injected_task=""):
        '''
        evaluation dataset initialization
        '''
        self.baseline = baseline
        self.target_task = target_task
        self.injected_task = injected_task

        config_path = os.path.join("dataset_config/{}.yaml".format(self.baseline))

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if self.baseline == "openprompt":
            self.target_task_config = config[self.baseline]["task_config"][self.target_task]
            self.injected_task_config = config[self.baseline]["task_config"][self.injected_task]
            

    def dataset_load_open_prompt(self, prompt_template, injected=True, injected_method='combine'):
        '''
        load dataset
        '''

        target_task = PI.create_task(open_config(config_path=f'{self.target_task_config}'), 100)
        instruction = target_task.get_instruction()

        if injected:
            inject_task = PI.create_task(open_config(config_path=f'{self.injected_task_config}'), 100, for_injection=True)
            attacker = PI.create_attacker(injected_method, inject_task)

        data = []

        for i, (data_prompt, ground_truth_label) in enumerate(target_task):
            if self.target_task=="alpaca":
                if injected:
                    instruction = data_prompt[0]
                    if injected_method=='fake_comp' or injected_method=='combine':
                        data_prompt = attacker.inject(data_prompt, i, target_task=target_task.task)
                    elif injected_method=='adaptive':
                        data_prompt = attacker.inject(data_prompt, i, target_task=target_task.task, defense_method=prompt_template)
                    else:
                        data_prompt = attacker.inject(data_prompt, i)
                else:
                    instruction = data_prompt[0]
                    data_prompt = data_prompt[1]
                item = data_preprocess(prompt_template, instruction, data_prompt, i)
                data.append(item)
            else:
                if injected:
                    if injected_method=='fake_comp' or injected_method=='combine':
                        data_prompt = attacker.inject(data_prompt, i, target_task=target_task.task)
                    elif injected_method=='adaptive':
                        data_prompt = attacker.inject(data_prompt, i, target_task=target_task.task, defense_method=prompt_template)
                    else:
                        data_prompt = attacker.inject(data_prompt, i)
                else:
                    data_prompt = data_prompt
                item = data_preprocess(prompt_template, instruction, data_prompt, i)
                data.append(item)

        return target_task, data


    def dataset_load_icl_instruction(self, demonstration, prompt_template, model="mistral", injected_method='combine'):
        '''
        load dataset
        '''
        inject_task = PI.create_icl_task(open_config(config_path=f'{self.injected_task_config}'), 100, for_injection=True)
        attacker = PI.create_attacker(injected_method, inject_task)

        data_injected = []

        for i, (data_prompt, ground_truth_label) in enumerate(inject_task):
            if injected_method=='fake_comp' or injected_method=='combine' or injected_method=='adaptive':
                data_prompt = attacker.inject(demonstration['input'], i, target_task='alpaca-demon')
            else:
                data_prompt = attacker.inject(demonstration['input'], i)
            data_injected.append(data_prompt)

        return inject_task, data_injected


    def dataset_load_icl(self, prompt_template, model="mistral", injected=True, injected_method='combine'):
        '''
        load dataset
        '''
        target_task = PI.create_icl_task(open_config(config_path=f'{self.target_task_config}'), 100)
        instruction = target_task.get_instruction()

        if injected:
            inject_task = PI.create_icl_task(open_config(config_path=f'{self.injected_task_config}'), 100, for_injection=True)
            attacker = PI.create_attacker(injected_method, inject_task)

        data_vallina = []
        data_injected = []

        for i, (data_prompt, ground_truth_label) in enumerate(target_task):
            data_vallina.append(data_prompt)
            if injected:
                if injected_method=='fake_comp' or injected_method=='combine' or injected_method=='adaptive':
                    data_prompt = attacker.inject(data_prompt, i, target_task=target_task.task)
                else:
                    data_prompt = attacker.inject(data_prompt, i)
                data_injected.append(data_prompt)

        return target_task, inject_task, instruction, data_vallina, data_injected