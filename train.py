from datasets import load_dataset,DatasetDict
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
from huggingface_hub import hf_hub_download
import os
from huggingface_hub import whoami



try:
    whoami()
    print("Successfully verified Hugging Face CLI login")
except Exception as e:
    raise ValueError("Please login using `huggingface-cli login`") from e

device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


# Load ChatAlpaca dataset
dataset = load_dataset('json', data_files='data/chatalpaca-10k.json')


dataset = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% validation

# work with the dictionary structure
train_test_dict = {
    'train': dataset['train'],
    'validation': dataset['test']  # Rename the split
dataset = DatasetDict(train_test_dict)

dataset['train'] = dataset['train'].select(range(500))
dataset['validation'] = dataset['validation'].select(range(50))  # Also limit validation set


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf") # Tokenize data for Llama3


tokenizer.pad_token = tokenizer.eos_token # Set padding token to be the same as EOS token

def tokenize_function(examples):
    # Process conversations into a format suitable for training
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for conversation in examples["conversations"]:
        # Combine human and gpt messages into a training format
        human_msgs = [turn["value"] for turn in conversation if turn["from"] == "human"]
        gpt_msgs = [turn["value"] for turn in conversation if turn["from"] == "gpt"]
        
        # take only the first pair from each conversation
        if human_msgs and gpt_msgs:  # take care of edge case
            full_text = f"Human: {human_msgs[0]}\nAssistant: {gpt_msgs[0]}"
            
            
            tokenized = tokenizer( # tokenize single example
                full_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            all_input_ids.append(tokenized["input_ids"][0])
            all_attention_mask.append(tokenized["attention_mask"][0])
            all_labels.append(tokenized["input_ids"][0].clone())
    
    return {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "labels": torch.stack(all_labels)
    }

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the base model 
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    max_memory={0: "8GB"}
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config) #redfine the model with out new config with peft 



training_args = TrainingArguments(
    output_dir="./llama2-chatalpaca",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)



trainer.train()


model.save_pretrained("./fine-tuned-llama3-chatalpaca")
tokenizer.save_pretrained("./fine-tuned-llama3-chatalpaca")
