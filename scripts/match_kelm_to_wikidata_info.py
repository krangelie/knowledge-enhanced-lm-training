import os
import json

from tqdm import tqdm
import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

"""This script allows to match all entities in the KELM Corpus (https://github.com/google-research-datasets/KELM-corpus) 
with another list of entities to add demographic information."""

@dataclass
class MyConfig:
    path_to_data_folder: str = "./data"
    wikidata_entity_infos: str = "wikidata_humans/wikidata_all_human_entities.jsonl"
    wikidata_gender_infos: str = "wikidata_humans/wikidata_genders.json"
    kelm_entity_infos: str = "kelm/entities.jsonl"
    output_dir: str = "wikidata_humans"


cs = ConfigStore.instance()

cs.store(name="conf", node=MyConfig())


@hydra.main(version_base=None, config_name="conf")
def main(cfg: MyConfig) -> None:

    out_dir = os.path.join(cfg.path_to_data_folder, cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load dict to map gender id to label
    print("Loading gender id-to-label map")
    with open(os.path.join(cfg.path_to_data_folder, cfg.wikidata_gender_infos), "r") as f:
        gender_map = json.load(f)["map"]

    # Load all KELM entities
    print("Loading KELM entities")
    with open(os.path.join(cfg.path_to_data_folder, cfg.kelm_entity_infos)) as json_file:
        kelm_ent_list = list(json_file)

    # Load previously extracted entity information (via `gather_entity_information_from_wikidata.py`)
    print("Load Wikidata entity information")
    ids = []
    genders = []
    dates_of_birth = []
    places_of_birth = []
    with open(os.path.join(cfg.path_to_data_folder, cfg.wikidata_entity_infos)) as json_file:
        ent_list = list(json_file)
        for entry in ent_list:
            entry = json.loads(entry)
            ids += [entry["entity_id"]]
            genders += [entry["gender"]]
            dates_of_birth += [entry["DOB"]]
            places_of_birth += [entry["POB"]]

    ids_set = set(ids)  # set for faster search

    # Match KELM to Wikidata info and save to new jsonl
    out_file_path = os.path.join(out_dir, "kelm_person_entities_matched_genders.jsonl")
    print("Match KELM with Wikidata information. Results saved in", out_file_path)
    with open(out_file_path, "w") as out_file:
        for i, ent in enumerate(tqdm(kelm_ent_list)):
            ent_dict = json.loads(ent)
            if ent_dict["id"].startswith("Q") and ent_dict["id"] in ids_set:
                match_idx = ids.index(ent_dict["id"])
                ent_dict["gender"] = gender_map[genders[match_idx]]
                ent_dict["dob"] = dates_of_birth[match_idx]
                ent_dict["pob"] = places_of_birth[match_idx]
                ent_dict["wikilist_match_idx"] = match_idx
                if i < 10:
                    print(ent_dict)
                out_file.write(json.dumps(ent_dict) + "\n")


if __name__ == '__main__':
    main()
