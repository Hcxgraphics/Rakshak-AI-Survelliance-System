import json

with open("path/to/instances_train2017.json") as f:
    data = json.load(f)

valid_annotations = []
for ann in data['annotations']:
    x, y, w, h = ann['bbox']
    if w > 1 and h > 1:
        valid_annotations.append(ann)

data['annotations'] = valid_annotations

with open("path/to/cleaned_instances_train2017.json", "w") as f:
    json.dump(data, f)
