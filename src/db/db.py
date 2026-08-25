from pymongo import MongoClient
import pandas as pd
import os
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from .fields import LIST_FIELDS

class DB:
    def __init__(self, uri: str=os.getenv("MONGO"), db_name="soot"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.materials = self.db["materials"]
        self.experiments = self.db["experiments"]

    def close(self):
        self.client.close()

    def generate_dataframes(self, materials_filter: Dict, experiments_filter: Dict, save_to_dir: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        materials = list(self.materials.find(materials_filter))
        dois = self.materials.distinct("_id", materials_filter)
        experiments_filter.update({"material_id": {"$in": dois}})
        experiments = list(self.experiments.find(experiments_filter))

        experiment_data = []
        for e in experiments:
            for i, temp in enumerate(e["temps"]):
                row = {field_name: field_value for field_name, field_value in e.items() if field_name not in LIST_FIELDS and field_name != "temps"}
                row["temperature"] = temp
                for field_name in LIST_FIELDS:
                    if e.get(field_name):
                        if isinstance(e[field_name], list) and len(e[field_name]) == len(e["temps"]):
                            row[field_name] = e[field_name][i]
                        else:
                            row[field_name] = e[field_name]
        
                experiment_data.append(row)

        mats = pd.DataFrame(materials)
        exps = pd.DataFrame(experiment_data)
        if save_to_dir is not None:
            save_to_dir = Path(save_to_dir)
            save_to_dir.mkdir(parents=True, exist_ok=True)
            mats.to_csv(save_to_dir / "materials.csv", index=False)
            exps.to_csv(save_to_dir / "experiments.csv", index=False)
        
        return mats, exps
