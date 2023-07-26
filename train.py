import os
import json
from tqdm import tqdm

from datasets import load_dataset
import neptune.new as neptune
from transformers.integrations import NeptuneCallback
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM,\
    Trainer, TrainingArguments, TextDataset, default_data_collator
from transformers.utils import logging
from sklearn.model_selection import train_test_split
from torch.cuda import is_available


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_text_files(data_json, dest_path, val_path=None):

    print("Building txt files.")
    with open(data_json) as json_file:
        json_list = list(json_file)

    def write_sentences(input_list, output_file):
        f = open(output_file, 'w')
        sentences = []
        for i, item in tqdm(enumerate(input_list)):
            #if i < 10:
            entry = json.loads(item)
            # filter out "??"-artifact
            if "⁇" not in entry["gen_sentence"]:
                sentences += [entry["gen_sentence"] + "\n"]
        f.writelines(sentences)
        return len(input_list), len(sentences)

    if val_path is not None:
        train_list, val_list = train_test_split(json_list, test_size=0.1, random_state=42)
        len_orig_data, len_filtered_data = write_sentences(train_list, dest_path)
        write_sentences(val_list, val_path)
    else:
        len_orig_data, len_filtered_data = write_sentences(json_list, dest_path)
    return len_orig_data, len_filtered_data



def train(cfg):
    train_path_txt = cfg.data.train_data_path.replace(".jsonl", "_train.txt")
    test_path_txt = cfg.data.test_data_path.replace(".jsonl", "_test.txt")

    len_orig_data_train, len_filtered_data_train = build_text_files(cfg.data.train_data_path, train_path_txt)
    len_orig_data_val, len_filtered_data_val = build_text_files(cfg.data.test_data_path, test_path_txt)

    dataset = load_dataset("text", data_files={"train": train_path_txt, "test": test_path_txt})
    print(dataset)
    print(dataset["train"][:5]["text"])

    if cfg.model.checkpoint:
        model_path = cfg.model.checkpoint
        print("Checkpoint:", model_path)
    else:
        model_path = cfg.model.model_name

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = 'left' # deteriorates performance... why?

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            #padding=True,
            truncation=True,
            max_length=cfg.model.context_len,
            #return_overflowing_tokens=True,
            #return_length=True,
        )
        return outputs

    tokenized_dataset = dataset.map(tokenize, batched=True)
    print(tokenized_dataset)
    print(tokenized_dataset["train"][:5]["input_ids"])

    if cfg.model.mlm:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, cfg.model.model_name, f"kelm_{cfg.data.dataset_version}"),
        learning_rate=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        overwrite_output_dir=True,
        num_train_epochs=cfg.model.epochs,
        per_device_train_batch_size=cfg.model.train_batch_size,
        per_device_eval_batch_size=cfg.model.eval_batch_size,
        eval_steps=10000,
        save_strategy="steps",
        save_steps=80000,
        warmup_steps=500,
        prediction_loss_only=True,
        evaluation_strategy="steps",
    )
    run = neptune.init_run()
    neptune_callback = NeptuneCallback(run=run)
    run["data_info/train_data"] = dataset
    run["data_info/tokenized_train_data"] = tokenized_dataset
    run["data_info/data_lengths"] = {"Num. KELM train cases": len_orig_data_train,
                                     "Num. KELM train cases (filtered)": len_filtered_data_train,
                                     "Num. KELM val cases": len_orig_data_val,
                                     "Num. KELM val cases (filtered)": len_filtered_data_val,
                                     "Removed cases train": len_orig_data_train - len_filtered_data_train,
                                     "Removed cases val": len_orig_data_val - len_filtered_data_val}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[neptune_callback],
    )

    logging.set_verbosity_info()
    if cfg.model.checkpoint:
        trainer.args.num_train_epochs = cfg.model.epochs
        trainer.train(cfg.model.checkpoint)
    else:
        train.train()
    #trainer.evaluate(eval_dataset=dataset["test"])
    trainer.save_model()
