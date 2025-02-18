from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback
)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 硬件优化配置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 模型配置
model_id = "/root/.cache/llama-3.1-8B-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="right",
    model_max_length=2048
)
tokenizer.pad_token = tokenizer.eos_token

# 回调系统
class TrainingMonitor(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and state.is_local_process_zero:
            memory = torch.cuda.memory_allocated() / 1e9
            print(f"Step {state.global_step} | GPU Memory: {memory:.2f}GB")

# 训练参数
args = TrainingArguments(
    output_dir="/root/vnm_train/tasks/llm/llama_fuse_sft/output/dense_sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    optim="adamw_torch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    bf16=True,
    learning_rate=5e-6,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    ddp_find_unused_parameters=False,
)

# 数据加载
train_dataset = load_dataset("json", 
    data_files="/root/vnm_train/tasks/llm/llama_fuse_sft/dataset/fuse_train_alpaca_format.json")['train']

test_dataset = load_dataset("json", 
    data_files="/root/vnm_train/tasks/llm/llama_fuse_sft/dataset/fuse_train_alpaca_format.json")['train'].select(range(10))

print(train_dataset[0])
print(f"train size: {len(train_dataset)}, test size: {len(test_dataset)}")
# 训练器配置
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=False,
    callbacks=[TrainingMonitor()]
)

# 启动训练
trainer.train()
trainer.save_model("final_model")
