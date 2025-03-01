from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your_model_name", device_map="auto")

from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model,
    checkpoint="path/to/pytorch_model_fsdp_0",
    device_map="auto",  # 自动分配设备
    offload_folder="offload",  # 可选：将部分参数卸载到CPU
    no_split_module_classes=["ModuleName"],  # 指定不分割的模块（如 Transformer 层）
)
