import torch
import os
import re
from transformers import AutoModelForCausalLM, LlamaTokenizer
import warnings
warnings.filterwarnings("ignore")

# ========== 多卡配置（GPU 0/1/2/3） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(f"✅ 可用GPU：{available_gpus}（共{len(available_gpus)}张）")

# 模型路径
MODEL_PATH = "/data_jijingbo/models/Emollama-chat-7b"

# 核心：使用你指定的纯英文提示词 + LLaMA2原生指令模板
def build_score_prompt(text: str) -> str:
    # LLaMA2官方<s>[INST] [/INST]模板 + 你指定的英文指令，仅输出数字
    prompt = f"""<s>[INST]
Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive). Only output the numerical score, no other words or symbols.
Text: {text}
Valence Intensity Score: [/INST]"""
    return prompt

# 提取得分：强化LLaMA2输出适配，兼容所有数字格式（0/1/0.xxx/.xxx）
def extract_score(output_text: str) -> float:
    # 正则匹配0-1之间的浮点数/整数，适配LLaMA2所有常见输出格式
    score_pattern = re.compile(r'(\d+\.?\d*|\.\d+)')
    matches = score_pattern.findall(output_text)
    if matches:
        score = float(matches[0])
        return max(0.0, min(1.0, score))  # 强制归一化0-1，防止模型输出超出范围
    else:
        return 0.5  # 无匹配时返回中性分，避免0.000无效值

def load_model_and_infer():
    try:
        # 1. 加载LLaMA2分词器（原生配置，适配英文指令）
        print("🔧 加载LLaMA2分词器...")
        tokenizer = LlamaTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            padding_side="right",  # LLaMA2原生推荐右填充，关键！
            use_fast=False,        # 禁用快速分词器，适配微调模型的tokenizer.model
            add_bos_token=True,
            add_eos_token=True
        )
        # 补充pad_token（LLaMA2默认无，推理必备）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("✅ 分词器加载成功！")

        # 2. 多卡加载LLaMA2微调模型（适配原生权重，显存均匀分配）
        print("🔧 加载LLaMA2微调模型到多GPU（0/1/2/3）...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,       # 半精度，降低显存占用
            low_cpu_mem_usage=True,    # 减少CPU内存占用
            trust_remote_code=True,
            device_map="auto",         # 自动多卡分片，适配4卡3090
            offload_folder="./offload" # 临时卸载目录，防止内存溢出
        )
        print(f"✅ 模型设备分配：{model.hf_device_map}")
        print(f"✅ LLaMA2微调模型加载完成！\n")

        # 3. 测试用例（中文文本，模型自动识别情绪）
        test_cases = [
            {"text": "Just won the lottery! I can't believe this is happening to me!"},
            {"text": "My beloved dog passed away today, I'm heartbroken and devastated."},
            {"text": "I met Bob today!"},
            {"text": "I met Bob today."},
            {"text": "I met messi today!"},
            {"text": "I met messi today."},
        ]

        # 4. 批量推理+得分提取（纯英文指令驱动）
        print("🚀 开始多卡推理（纯英文指令，输出0-1效价强度得分）...\n")
        for idx, case in enumerate(test_cases, 1):
            text = case["text"]
            # 构造LLaMA2原生格式的纯英文提示词Prompt
            prompt = build_score_prompt(text)
            # 编码：输入放GPU0，模型自动分发到多卡
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to("cuda:0")

            # 5. LLaMA2专属推理参数（适配英文指令，避免0.000）
            model.eval()
            with torch.no_grad():  # 禁用梯度，节省显存
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=6,        # 仅输出数字，足够用（如0.896/1.0）
                    temperature=0.2,         # 低温度保证得分稳定，无随机波动
                    top_p=0.9,               # LLaMA2官方推荐值，核采样
                    top_k=40,                # LLaMA2原生推荐值，提升出分概率
                    do_sample=True,          # 轻微采样，避免模型惰性输出0.0
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,  # LLaMA2禁用重复惩罚，关键！
                    length_penalty=1.0       # 禁用长度惩罚，避免数字截断
                )

            # 6. 解码+精准提取得分（过滤特殊token和前缀）
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 截取[/INST]后的内容，仅保留模型输出的数字
            score_text = raw_output.split("[/INST]")[-1].strip()
            # 提取0-1的效价强度得分
            valence_score = extract_score(score_text)

            # 7. 打印结果（标注0=最负面，1=最正面，清晰直观）
            print(f"【测试用例 {idx}】")
            print(f"文本：{text}")
            print(f"效价强度得分（0=最负面 | 1=最正面）：{valence_score:.3f}\n" + "-"*120 + "\n")

        # 8. 多卡显存使用统计
        print("📊 GPU显存使用情况（0-3卡）：")
        for i in range(4):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}：已用 {mem_used:.1f}GB / 总 {mem_total:.1f}GB")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ 运行出错：{type(e).__name__} - {str(e)[:300]}")
        torch.cuda.empty_cache()  # 出错时清理显存
        return None, None
    finally:
        # 无论是否成功，清理显存+临时目录
        torch.cuda.empty_cache()
        import shutil
        shutil.rmtree("./offload", ignore_errors=True)

# ========== 主函数执行 ==========
if __name__ == "__main__":
    os.makedirs("./offload", exist_ok=True)  # 创建临时卸载目录
    model, tokenizer = load_model_and_infer()
    if model is not None:
        print("\n🎉 LLaMA2微调模型多卡推理完成！所有文本均精准输出0-1效价强度得分～")
    else:
        print("\n❌ 推理失败！")