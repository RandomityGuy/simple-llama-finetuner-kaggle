import os
import gc
import argparse
import random
import torch
import transformers
import peft
import datasets
import gradio as gr
import json
import sys

model = None
tokenizer = None
current_peft_model = None

def load_base_model():
    global model
    print('Loading base model...')
    model = transformers.LlamaForCausalLM.from_pretrained(
        'decapoda-research/llama-7b-hf',
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={'':0}
    )

def load_tokenizer():
    global tokenizer
    print('Loading tokenizer...')
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        'decapoda-research/llama-7b-hf',
    )

def load_peft_model(model_name):
    global model
    print('Loading peft model ' + model_name + '...')
    model = peft.PeftModel.from_pretrained(
        model, model_name,
        torch_dtype=torch.float16
    )

def reset_model():
    global model
    global tokenizer
    global current_peft_model

    del model
    del tokenizer

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    model = None
    tokenizer = None
    current_peft_model = None

def generate_text(
    peft_model,
    text, 
    temperature, 
    top_p, 
    top_k, 
    repetition_penalty, 
    max_new_tokens,
    progress=gr.Progress(track_tqdm=True)
):
    global model
    global tokenizer
    global current_peft_model

    if (peft_model == 'None'): peft_model = None

    if (current_peft_model != peft_model):
        if (current_peft_model is None):
            if (model is None): load_base_model()
        else:
            reset_model()
            load_base_model()
            load_tokenizer()

        current_peft_model = peft_model
        if (peft_model is not None):
            load_peft_model(peft_model)

    if (model is None): load_base_model()
    if (tokenizer is None): load_tokenizer()

    assert model is not None
    assert tokenizer is not None

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    generation_config = transformers.GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_beams=1,
    )

    with torch.no_grad():
        output = model.generate(  # type: ignore
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config
        )[0].cuda()

    return tokenizer.decode(output, skip_special_tokens=True).strip()

def tokenize_and_train(
    training_text,
    micro_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    max_samples = 2000,
    max_seq_length=512,
    epochs=1,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.01,
):
    global model
    global tokenizer

    if (model is None): load_base_model()
    if (tokenizer is None): 
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "decapoda-research/llama-7b-hf", add_eos_token=True
        )

    assert model is not None
    assert tokenizer is not None

    tokenizer.pad_token_id = 0

    paragraphs = training_text
    paragraphs = [x.strip() for x in paragraphs]

    print("Number of samples: " + str(len(paragraphs)))
        
    def tokenize(item):
        assert tokenizer is not None
        result = tokenizer(
            item["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    def to_dict(text):
        return {"text": text}

    paragraphs = [to_dict(x) for x in paragraphs]
    data = datasets.Dataset.from_list(paragraphs)
    data = data.shuffle().map(lambda x: tokenize(x))

    model = peft.prepare_model_for_int8_training(model)

    model = peft.get_peft_model(model, peft.LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    ))

    output_dir = f"lora-tuned"

    print("Training...")

    training_args = transformers.TrainingArguments(
        # Set the batch size for training on each device (GPU, CPU, or TPU).
        per_device_train_batch_size=micro_batch_size, 

        # Number of steps for gradient accumulation. This is useful when the total 
        # batch size is too large to fit in GPU memory. The effective batch size 
        # will be the product of 'per_device_train_batch_size' and 'gradient_accumulation_steps'.
        gradient_accumulation_steps=gradient_accumulation_steps,  

        # Number of warmup steps for the learning rate scheduler. During these steps, 
        # the learning rate increases linearly from 0 to its initial value. Warmup helps
        #  to reduce the risk of very large gradients at the beginning of training, 
        # which could destabilize the model.
        # warmup_steps=100, 

        # The total number of training steps. The training process will end once this 
        # number is reached, even if not all the training epochs are completed.
        max_steps=max_samples, 

        # The total number of epochs (complete passes through the training data) 
        # to perform during the training process.
        num_train_epochs=epochs,  

        # The initial learning rate to be used during training.
        learning_rate=learning_rate, 

        # Enables mixed precision training using 16-bit floating point numbers (FP16). 
        # This can speed up training and reduce GPU memory consumption without 
        # sacrificing too much model accuracy.
        fp16=True,  

        # The frequency (in terms of steps) of logging training metrics and statistics 
        # like loss, learning rate, etc. In this case, it logs after every 20 steps.
        logging_steps=20, 

        # The output directory where the trained model, checkpoints, 
        # and other training artifacts will be saved.
        output_dir=output_dir, 

        # The maximum number of checkpoints to keep. When this limit is reached, 
        # the oldest checkpoint will be deleted to save a new one. In this case, 
        # a maximum of 3 checkpoints will be kept.
        save_total_limit=1,  
    )


    trainer = transformers.Trainer(
        # The pre-trained model that you want to fine-tune or train from scratch. 
        # 'model' should be an instance of a Hugging Face Transformer model, such as BERT, GPT-2, T5, etc.
        model=model, 

        # The dataset to be used for training. 'data' should be a PyTorch Dataset or 
        # a compatible format, containing the input samples and labels or masks (if required).
        train_dataset=data, 

        # The TrainingArguments instance created earlier, which contains various 
        # hyperparameters and configurations for the training process.
        args=training_args, 

        # A callable that takes a batch of samples and returns a batch of inputs for the model. 
        # This is used to prepare the input samples for training by batching, padding, and possibly masking.
        data_collator=transformers.DataCollatorForLanguageModeling( 
            tokenizer,  
            # Whether to use masked language modeling (MLM) during training. 
            # MLM is a training technique used in models like BERT, where some tokens in the 
            # input are replaced by a mask token, and the model tries to predict the 
            # original tokens. In this case, MLM is set to False, indicating that it will not be used.
            mlm=False, 
        ),
    )

    model.config.use_cache = False
    result = trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(output_dir)

    del data
    reset_model()

    return result

def random_hyphenated_word():
    word_list = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']
    word1 = random.choice(word_list)
    word2 = random.choice(word_list)
    return word1 + '-' + word2

if __name__ == '__main__':

    trainmsgs = None
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        trainmsgs = json.load(f)
        
    prepared_msgs = []
    for msg in trainmsgs["messages"]:
        prepared_msgs.append(msg)

    mbs = 1
    if len(sys.argv) > 2:
        mbs = int(sys.argv[2])

    gas = 1
    if len(sys.argv) > 3:
        gas = int(sys.argv[3])

    lr = 3e-4
    if len(sys.argv) > 4:
        lr = float(sys.argv[4])

    ms = 2000
    if len(sys.argv) > 5:
        ms = int(sys.argv[5])

    tokenize_and_train(prepared_msgs, mbs, gas, lr, ms)