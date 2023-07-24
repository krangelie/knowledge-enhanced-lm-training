import os
from datetime import datetime

from transformers import pipeline, AutoTokenizer
import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


"""Generate texts with GPT-2 from user prompts via terminal."""


@dataclass
class MyConfig:
    model_name: str = "gpt2-medium"
    language_model_path: str = "/export/home/kraft/data/kelm/output/gpt2-medium/kelm_full"
    output_dir: str = "/export/home/kraft/data/kelm/output/gpt2-medium/kelm_full/generated_texts_for_eval" #"/export/home/kraft/data/kelm/output/gpt2/kelm_quarter/checkpoint-61043/generated_texts_for_eval"
    tokenizer: str = model_name
    num_return_sequences: int = 3


cs = ConfigStore.instance()
cs.store(name="conf", node=MyConfig())


@hydra.main(version_base=None, config_name="conf")
def run_language_modeling(cfg: MyConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    mask_filler_pipeline = pipeline("text-generation", model=cfg.language_model_path, tokenizer=tokenizer)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #if cfg.mode.inference_mode == "file":
    #    _apply_to_file(mask_filler_pipeline, cfg.mode.input_path, output_path)
    #elif cfg.mode.inference_mode == "interactive":
    _apply_to_user_input(mask_filler_pipeline, cfg.model_name, cfg.num_return_sequences, output_dir)
    #else:
    #    sys.exit("Aborting - Please specify inference_mode 'file' or 'interactive'.")


def _apply_to_user_input(mask_filler_pipeline, model_name, num_return_sequences, output_dir):
    timestamp = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    out_file = os.path.join(output_dir, f"{model_name}-inferred_from_ui.tsv")
    print(f"--- Text generation with {model_name} ---")
    print("Type in any prompt and then press ENTER.")
    print("Enter Q to quit.")
    while True:
        with open(out_file, "a") as o:
            user_input = input("Input: ")
            if user_input == "Q":
                print("Closing UI. Results were stored in", out_file)
                break
            model_output = mask_filler_pipeline(user_input, num_return_sequences=num_return_sequences)

            top_outputs = "----" + model_name + "\t" + timestamp + "----\n"
            print("Output:")
            for i, output in enumerate(model_output):
                print(f"{i+1}.: {output['generated_text']}")
                top_outputs += output['generated_text'] + "\n"
            print()

            o.write(top_outputs)


if __name__ == "__main__":
    run_language_modeling()
