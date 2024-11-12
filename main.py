from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
from huggingface_hub import hf_hub_download
import os

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


# Load ChatAlpaca dataset, assuming it's formatted as JSON or similar
dataset = load_dataset('json', data_files='data/chatalpaca-10k.json')


dataset['train'] = dataset['train'].select(range(500))

# Split dataset into train and validation
dataset = dataset['train'].train_test_split(test_size=0.1)
dataset['validation'] = dataset['test']
del dataset['test']



# Tokenize data for Llama3
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def tokenize_function(examples):
    # Process conversations into a format suitable for training
    conversations = examples["conversations"]
    full_texts = []
    
    # Process each conversation
    for conv in conversations:
        # Combine human and gpt messages into a training format
        human_msgs = [turn["value"] for turn in conv if turn["from"] == "human"]
        gpt_msgs = [turn["value"] for turn in conv if turn["from"] == "gpt"]
        
        # Combine into conversation pairs
        for human, gpt in zip(human_msgs, gpt_msgs):
            full_text = f"Human: {human}\nAssistant: {gpt}"
            full_texts.append(full_text)
    
    return tokenizer(
        full_texts, 
        padding="max_length", 
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_labels=True
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset into train and validation
dataset = dataset['train'].train_test_split(test_size=0.1)
dataset['validation'] = dataset['test']
del dataset['test']

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the base Llama 2-7b model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)



training_args = TrainingArguments(
    output_dir="./llama2-chatalpaca",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
