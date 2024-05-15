from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import bentoml


def generate_response(model, tokenizer, text: str = "what is the result of doing 2+2") -> str:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=680)

    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if decoded_text.lower().startswith(text.lower()):
        decoded_text = decoded_text[len(text):].strip()

    return f"Question: {text} - Answer: {decoded_text}"


def export_model(model, tokenizer: AutoTokenizer) -> bool:
    """
    :param model:
    :param tokenizer:
    :return: success: bool to indicate if model export was successful or not
    """
    try:
        bentoml.pytorch.save_model(
            "demo_mnist",  # Model name in the local Model Store
            model,  # Model instance being saved
            labels={  # User-defined labels for managing models in BentoCloud
                "owner": "cs",
                "stage": "test",
            },
            custom_objects={  # Save additional user-defined Python objects
                "tokenizer": tokenizer,
                "model": model
            }
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        return False
    return True


# fine-tuned model id
model_id = f"mistral-7b-style/checkpoint-18"

# load base LLM model, LoRA params and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = '''
'''

print(generate_response(model, tokenizer, text=text))
