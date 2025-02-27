import sys
import yaml
from pathlib import Path
from transformers import HfArgumentParser

# 原数据类定义保持不变
@dataclass
class ScriptArguments:
    # ... [保留原有字段定义] ...

def parse_yaml_config(file_path: str) -> ScriptArguments:
    """从YAML文件加载参数并映射到数据类"""
    with open(file_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    parser = HfArgumentParser(ScriptArguments)
    return parser.parse_dict(yaml_config)[0]

def main():
    # 检查是否存在YAML文件参数
    if len(sys.argv) > 1 and Path(sys.argv[1]).suffix in (".yaml", ".yml"):
        yaml_path = sys.argv[1]
        script_args = parse_yaml_config(yaml_path)
        sys.argv.pop(1)  # 移除已处理的YAML参数
    else:
        parser = HfArgumentParser(ScriptArguments)
        script_args = parser.parse_args_into_dataclasses()[0]
    
    # 后续训练代码保持不变
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        learning_rate=script_args.learning_rate,
        # ... [其他参数赋值] ...
    )
    # ... [数据集加载、模型初始化等] ...

if __name__ == "__main__":
    main()
