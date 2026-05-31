# 3D-GAN Structured Dashboard

이 폴더를 GitHub Pages에 그대로 올리면 `index.html`이 대시보드로 보입니다.

## 실제 Structured 결과 반영 순서

1. `Structured/Pipeline.py`를 실행하고 반환 결과를 pickle로 저장합니다.
2. `tools/export_dashboard_results.py`로 `structured-results.js`를 생성합니다.
3. `index.html`과 `structured-results.js`를 GitHub에 push합니다.

예시:

```python
# save_pipeline_result.py
import copy
import pickle
from Structured.Pipeline import DEFAULT_CONFIG, run_pipeline

cfg = copy.deepcopy(DEFAULT_CONFIG)
cfg["data_path"] = "./data/features.csv"
cfg["label_path"] = "./data/labels.csv"
cfg["label_groups"] = [[0], [1, 2], [3, 4, 5, 6, 7], [8, 9]]

result = run_pipeline(cfg)
with open("output/results.pkl", "wb") as f:
    pickle.dump(result, f)
```

```bash
python save_pipeline_result.py
python tools/export_dashboard_results.py --input output/results.pkl --out structured-results.js
```
