class Qwen2PromptTemplate:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def build_prompt(self, user_message):
        if self.system_prompt is not None:
            SYS = f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
        else:
            SYS = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        CONVO = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        return SYS + CONVO

class GemmaPromptTemplate:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def build_prompt(self, user_message):
        return f"""<bos><start_of_turn>user
{self.system_prompt}

{user_message}<end_of_turn>
<start_of_turn>model\n"""
