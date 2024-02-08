"""Custom PEFT model for full finetuning a transformer model with a learned context(prompt)"""

from peft import PeftModelForCausalLM
from peft.config import PeftConfig
from peft.utils import PeftType, TaskType
from transformers import PreTrainedModel
from peft.tuners import PromptEmbedding
from peft.mapping import _prepare_prompt_learning_config
import torch


class FullTuningWithPromptForCausalLM(PeftModelForCausalLM):
    """PEFT model that combines a learned soft prompt and full finetuning together."""
    
    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            if isinstance(module, PreTrainedModel):
                # do not freeze the Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

            if value.shape[0] == self.base_model.config.vocab_size or (
                deepspeed_distributed_tensor_shape is not None
                and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
            ):
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        else:
            raise ValueError("Not supported")

        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

        # freeze the prompt encoder when full finetuning
        for param in prompt_encoder.parameters():
            param.requires_grad = False
            
    def save_pretrained(self, *args, **kwargs):
        # save the full model instead of the adapter
        self.base_model.save_pretrained(*args, **kwargs)


def get_full_tuning_with_prompt_model(                # modded from `get_peft_model` in PEFT
    model: PreTrainedModel, peft_config: PeftConfig, load_prompt=None
):
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    model = FullTuningWithPromptForCausalLM(model, peft_config)
    
    if load_prompt is not None:
        model.load_adapter(load_prompt, "default")
    
    return model