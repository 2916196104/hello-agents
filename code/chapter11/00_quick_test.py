"""用真实 GSM8K 数据和 Qwen 模型验证 SFT、GRPO 各一步训练及 LoRA 保存。

这是流程测试，不用于评估模型效果。CPU 使用 FP32，首次运行需要下载模型和数据集。
安装依赖：python -m pip install -r code/chapter11/requirements.txt
"""

import gc
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from hello_agents.rl import create_accuracy_reward, create_rl_dataset, create_sft_dataset
from hello_agents.tools import RLTrainingTool


MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "quick_test"


def run_tool(tool: RLTrainingTool, parameters: dict) -> dict:
    """工具会把异常转成 JSON，必须检查状态才能判定通过。"""
    result = json.loads(tool.run(parameters))
    if result.get("status") != "success":
        raise RuntimeError(result.get("traceback") or result.get("message") or str(result))
    return result


def train_once(algorithm: str, output_dir: Path) -> dict:
    """直接传递 TRL 配置，避免旧版工具忽略 LoRA 和生成长度参数。"""
    set_seed(42)
    use_cpu = not torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
    )
    common: dict[str, Any] = dict(
        output_dir=str(output_dir),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
        use_cpu=use_cpu,
        dataloader_pin_memory=not use_cpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
    )
    trainer: SFTTrainer | GRPOTrainer
    if algorithm == "sft":
        dataset = create_sft_dataset(max_samples=2)
        # 只保留 text，按完整样本计算损失，避免截断后全部标签被屏蔽。
        dataset = dataset.select_columns(["text"])
        sft_args = SFTConfig(**common, max_length=256, completion_only_loss=False)
        trainer = SFTTrainer(
            model=model, args=sft_args, train_dataset=dataset,
            processing_class=tokenizer, peft_config=lora_config,
        )
    elif algorithm == "grpo":
        dataset = create_rl_dataset(max_samples=2, model_name=MODEL_NAME)
        grpo_args = GRPOConfig(
            **common, num_generations=2, generation_batch_size=2,
            max_prompt_length=256, max_completion_length=32,
            beta=0.0, use_vllm=False,
        )
        trainer = GRPOTrainer(
            model=model, args=grpo_args, train_dataset=dataset,
            reward_funcs=create_accuracy_reward(),
            processing_class=tokenizer, peft_config=lora_config,
        )
    else:
        raise ValueError(f"不支持的训练算法: {algorithm}")

    if not isinstance(trainer.model, PeftModel):
        raise RuntimeError("LoRA 模型未创建，停止测试以避免全参数训练")
    trainable, total = trainer.model.get_nb_trainable_parameters()
    if not 0 < trainable < total:
        raise RuntimeError("LoRA 未生效，停止测试以避免全参数训练")
    print(f"LoRA 可训练参数: {trainable:,}/{total:,}", flush=True)
    output = trainer.train()
    if output.global_step != 1 or not math.isfinite(output.training_loss):
        raise RuntimeError(f"训练未正常完成: {output}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()
    for filename in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
        if not (output_dir / filename).is_file():
            raise RuntimeError(f"缺少训练输出文件: {output_dir / filename}")
    result = {
        "status": "success", "algorithm": algorithm.upper(),
        "global_step": output.global_step, "loss": output.training_loss,
        "trainable_parameters": trainable, "output_dir": str(output_dir),
    }
    # 两个训练器顺序释放，避免 CPU 内存中同时保留两份模型。
    del trainer, model, output
    gc.collect()
    if not use_cpu:
        torch.cuda.empty_cache()
    return result


def quick_test() -> None:
    if version("trl") != "0.24.0":
        raise RuntimeError("请先安装本章依赖：python -m pip install -r code/chapter11/requirements.txt")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    print("=" * 80)
    print(f"快速实验测试（设备：{'GPU' if torch.cuda.is_available() else 'CPU'}）")
    print("SFT 和 GRPO 各执行一步真实训练；不测试数学准确率是否提高。")
    print("=" * 80, flush=True)
    tool = RLTrainingTool()

    print("\n测试1: 数据加载", flush=True)
    data = run_tool(tool, {"action": "load_dataset", "format": "sft", "max_samples": 5})
    if data.get("dataset_size") != 5:
        raise RuntimeError(f"数据集样本数不符: {data}")
    print("✅ 数据加载通过: 5 个样本", flush=True)

    for index, algorithm in enumerate(("sft", "grpo"), start=2):
        print(f"\n测试{index}: {algorithm.upper()} 训练和保存", flush=True)
        result = train_once(algorithm, OUTPUT_ROOT / algorithm)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"✅ {algorithm.upper()} 训练和保存通过", flush=True)

    print("\n测试4: 奖励函数", flush=True)
    run_tool(tool, {"action": "create_reward", "reward_type": "accuracy"})
    rewards = create_accuracy_reward()(
        ["Final Answer: 2", "Final Answer: 3"], ground_truth=["2", "2"],
    )
    if rewards != [1.0, 0.0]:
        raise RuntimeError(f"奖励函数结果不符: {rewards}")
    print("✅ 奖励函数通过: 正确答案 1.0，错误答案 0.0")
    print("\n✅ 所有测试通过！")
    print(f"LoRA 适配器保存目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    quick_test()
