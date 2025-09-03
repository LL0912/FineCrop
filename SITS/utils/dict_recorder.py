import os
import numpy as np
from openpyxl import Workbook
import pandas as pd
class DictRecorder:
    def __init__(self):
        self._records = {}

    def update_dict(self, new_dict):
        for key,value in new_dict.items():
            if key in self._records:
                self._records[key].append(value)
            else:
                self._records[key] = [value]

    def get_dict_mean(self):
        mean_dict = {}
        for key, value in self._records.items():
            mean_dict[key] = sum(value) / len(value)
        return mean_dict

    def get_records(self):
        return self._records

    def save_dict_to_excel(self, save_path):
        if os.path.exists(save_path):
            os.remove(save_path)
        workbook = Workbook()
        # 写入的时候包括编码，名称，个数
        for k, v in self._records.items():
            sheet = workbook.create_sheet(title=k)
            if isinstance(v, float):
                v = np.asarray([v]).reshape((-1, 1))
            else:
                v=np.asarray(v)
                if v.ndim > 2:
                    s=v.squeeze().shape[-1]
                    v=v.reshape((-1,s))
                elif v.ndim == 2:
                    v = v
                elif v.ndim == 1:
                    v = v.reshape((-1, 1))
            df = pd.DataFrame(v)
            for row in df.iterrows():
                sheet.append(row[1].tolist())
            
        workbook.remove(workbook['Sheet'])
        workbook.save(save_path)

    def save_dict_to_npy(self, save_path):
        np.save(save_path, **self._records)

    def load_dict_from_npy(self, save_path):
        data = np.load(save_path)
        self._records = {key: data[key] for key in data.files}
        return self._records
