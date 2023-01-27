import os
import json


from datasets import load_dataset
import neptune.new as neptune
from transformers.integrations import NeptuneCallback
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset
from sklearn.model_selection import train_test_split


def build_text_files(data_json, dest_path, val_path=None):
    print("Building txt files.")
    with open(data_json) as json_file:
        json_list = list(json_file)

    def write_sentences(input_list, output_file):
        f = open(output_file, 'w')
        sentences = []
        for item in input_list:
            entry = json.loads(item)
            sentences += [entry["gen_sentence"] + "\n"]
        f.writelines(sentences)

    if val_path is not None:
        train_list, val_list = train_test_split(json_list, test_size=0.1, random_state=42)
        write_sentences(train_list, dest_path)
        write_sentences(val_list, val_path)
    else:
        write_sentences(json_list, dest_path)


def train(cfg):
    train_path_txt = cfg.data.train_data_path.replace(".jsonl", "_train.txt")
    #val_path_txt = cfg.data.train_data_path.replace(".jsonl", "_val.txt")
    test_path_txt = cfg.data.test_data_path.replace(".jsonl", "_test.txt")

    build_text_files(cfg.data.train_data_path, train_path_txt)
    build_text_files(cfg.data.test_data_path, test_path_txt)

    dataset = load_dataset("text", data_files={"train": train_path_txt, "test": test_path_txt})
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=cfg.model.context_len,
            #return_overflowing_tokens=True,
            #return_length=True,
        )
        return outputs

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path)
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, cfg.model.model_name_or_path, f"kelm_{cfg.data.dataset_version}"),
        overwrite_output_dir=True,
        num_train_epochs=cfg.model.epochs,
        per_device_train_batch_size=cfg.model.train_batch_size,
        per_device_eval_batch_size=cfg.model.eval_batch_size,
        eval_steps=400,
        save_strategy="epoch",
        warmup_steps=500,
        prediction_loss_only=True,
        evaluation_strategy="steps",
    )

    run = neptune.init_run(project='kraft-ml/KELM')
    neptune_callback = NeptuneCallback(run=run)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[neptune_callback],
    )

    trainer.train()
    trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    trainer.save_model()
