# Data

The notebooks expect one CSV file, `multiclass_split.csv` (it is not included in this repo). On first run, the data cell asks you to upload it and saves it to Google Drive so all four notebooks use the same file.

It holds 41,838 Stack Exchange questions, each with exactly one of ten tags, with the train / validation / test split already assigned. It is the same file as in [part 1](https://github.com/rajatsxngh/modernbert-vs-qwen3-qlora).

| Column | Description |
|---|---|
| `id` | Stack Exchange question id |
| `text` | Title and body joined by a space; the model input (raw HTML and code kept, not cleaned) |
| `label` | Tag as a class index 0–9 |
| `tag` | Tag name, used as the answer the model writes |
| `split` | `train` (29,287), `val` (6,276) or `test` (6,275) |
| `decoder_train` | 1 for the fixed subset of 5,000 training questions |
| `val_subset` | 1 for the fixed subset of 1,000 validation questions used to monitor training |

Tag names in label order:

```python
class_names = df.drop_duplicates('label').sort_values('label')['tag'].tolist()
# ['c#', 'java', 'php', 'javascript', 'android', 'jquery', 'c++', 'python', 'iphone', 'asp.net']
```
