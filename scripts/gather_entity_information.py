from gzip import GzipFile
import os
import json

from tqdm import tqdm
import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


def filter_claims(entity_object):
    try:
        instance_of = entity_object["claims"]["P31"][0]["mainsnak"]["datavalue"]["value"]["id"]
    except:
        return None

    if instance_of == "Q5":  #check if human
        person_entity = {"entity_id": entity_object["id"], "gender": "", "DOB": "", "POB": ""}
        try:
            person_entity["gender"] = entity_object["claims"]["P21"][0]["mainsnak"]["datavalue"]["value"]["id"]
        except:
            pass
        try:
            person_entity["DOB"] = entity_object["claims"]["P569"][0]["mainsnak"]["datavalue"]["value"]["time"]
        except:
            pass
        try:
            person_entity["POB"] = entity_object["claims"]["P19"][0]["mainsnak"]["datavalue"]["value"]["id"]
        except:
            pass
        return person_entity
    else:
        return None


@dataclass
class MyConfig:
    wikidata_path: str = "./data/latest-all.json.gz"
    output_dir: str = "./wikidata_humans"


cs = ConfigStore.instance()
cs.store(name="conf", node=MyConfig())


@hydra.main(version_base=None, config_name="conf")
def main(cfg: MyConfig) -> None:

    wikidata_path = cfg.wikidata_path
    out_dir = cfg.output_dir
    with GzipFile(wikidata_path) as gf:
        for ln in tqdm(gf):
            if ln == b'[\n' or ln == b']\n':
                continue
            if ln.endswith(b',\n'):  # all but the last element
                obj = json.loads(ln[:-2])
            else:
                obj = json.loads(ln)
            entity_information = filter_claims(obj)
            if entity_information is not None:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "wikidata_all_human_entities.jsonl"), "a") as f:
                    f.write(json.dumps(entity_information) + "\n")


if __name__ == '__main__':
    main()
