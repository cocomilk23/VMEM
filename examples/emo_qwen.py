import torch
import os
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ========== 多卡配置 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(f"✅ 可用GPU：{available_gpus}（共{len(available_gpus)}张）")

# ===== 本地模型路径 =====
MODEL_PATH = "/data_jijingbo/models/Qwen2.5-7B-Instruct-128K"

# ===== Qwen2.5 官方 chat prompt =====
def build_score_messages(text: str):
    return [
        {
            "role": "system",
            "content": (
                "You are a professional sentiment analysis system. "
                "Your task is to evaluate emotional valence intensity."
            )
        },
        {
            "role": "user",
            "content": (
                "Score the emotional valence of the following text on a scale from 0 to 1.\n"
                "0 = extremely negative\n"
                "0.5 = neutral\n"
                "1 = extremely positive\n\n"
                "IMPORTANT:\n"
                "- Output ONLY one number between 0 and 1.\n"
                "- Do NOT output words, symbols, or explanations.\n\n"
                f"Text:\n{text}\n\nScore:"
            )
        }
    ]

# ===== 提取 0–1 数值 =====
def extract_score(output_text: str) -> float:
    score_pattern = re.compile(r'(\d+\.?\d*|\.\d+)')
    matches = score_pattern.findall(output_text)
    if matches:
        score = float(matches[0])
        return max(0.0, min(1.0, score))
    else:
        return 0.5  # fallback：中性

def load_model_and_infer():
    try:
        # 1️⃣ tokenizer（明确本地）
        print("🔧 加载 Qwen2.5 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right"
        )
        print("✅ tokenizer 加载成功")

        # 2️⃣ 模型（多卡）
        print("🔧 加载 Qwen2.5-7B-Instruct 到多 GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_folder="./offload_qwen"
        )
        model.eval()
        print(f"✅ 模型设备分配：{model.hf_device_map}")

        # 3️⃣ 测试样例（与 Emollama 对照一致）
        test_cases = [
            "Just won the lottery! I can't believe this is happening to me!",
            "My beloved dog passed away today, I'm heartbroken and devastated.",
            "Got accepted into my dream university with a full scholarship!",
            "Stuck in traffic for hours, I feel exhausted and annoyed.",
            "Finally achieved my fitness goal after months of hard work!",
            "Lost my job unexpectedly, uncertain about the future now."
        ]

        print("\n🚀 开始 Qwen2.5 推理（0–1 效价强度）...\n")

        for idx, text in enumerate(test_cases, 1):
            messages = build_score_messages(text)

            # 关键：Qwen 官方 chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=6,
                    temperature=0.3,   # 防止数值塌缩
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            score_text = decoded.split("Score:")[-1].strip()
            valence_score = extract_score(score_text)

            print(f"【测试用例 {idx}】")
            print(f"文本：{text}")
            print(f"Qwen2.5 效价强度得分（0=最负面 | 1=最正面）：{valence_score:.3f}")
            print("-" * 120)

        # 4️⃣ 显存统计
        print("\n📊 GPU 显存使用情况：")
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {used:.1f}GB / {total:.1f}GB")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ 出错：{type(e).__name__} - {str(e)[:300]}")
        torch.cuda.empty_cache()
        return None, None

    finally:
        torch.cuda.empty_cache()
        import shutil
        shutil.rmtree("./offload_qwen", ignore_errors=True)

# ========== 主函数 ==========
if __name__ == "__main__":
    os.makedirs("./offload_qwen", exist_ok=True)
    model, tokenizer = load_model_and_infer()
    if model is not None:
        print("\n🎉 Qwen2.5 对照实验完成！")
    else:
        print("\n❌ Qwen2.5 推理失败")
