from collections import Counter
from pathlib import Path
import json
import sys

# Get dataset path from command line or use default
if len(sys.argv) > 1:
    dataset_base = sys.argv[1]
else:
    dataset_base = r'C:\workspace\data\.cache\datasets\97ff5345-9d55-49af-9724-b6b1158569ae'

train_txt = Path(dataset_base + '_yolo') / 'train.txt'
val_txt = Path(dataset_base + '_yolo') / 'val.txt'

print(f"Analyzing: {dataset_base}")
print(f"Train file: {train_txt}")
print(f"Val file: {val_txt}")
print()

# Count class distribution in train set
train_classes = []
with open(train_txt, 'r') as f:
    train_images = [line.strip() for line in f]

for img_path in train_images:
    label_path = Path(img_path.replace('\\images\\', '\\labels\\').replace('.jpg', '.txt'))
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    train_classes.append(class_id)

# Count class distribution in val set
val_classes = []
with open(val_txt, 'r') as f:
    val_images = [line.strip() for line in f]

for img_path in val_images:
    label_path = Path(img_path.replace('\\images\\', '\\labels\\').replace('.jpg', '.txt'))
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    val_classes.append(class_id)

train_dist = Counter(train_classes)
val_dist = Counter(val_classes)

print(f'Train set: {len(train_images)} images, {len(train_classes)} objects')
print(f'Train class distribution (top 15):')
for class_id, count in sorted(train_dist.items(), key=lambda x: -x[1])[:15]:
    print(f'  Class {class_id:2d}: {count:3d} objects')

print(f'\nVal set: {len(val_images)} images, {len(val_classes)} objects')
print(f'Val class distribution:')
for class_id, count in sorted(val_dist.items()):
    print(f'  Class {class_id:2d}: {count:2d} objects')

print(f'\nClasses in train: {len(train_dist)}')
print(f'Classes in val: {len(val_dist)}')
print(f'Classes with NO training examples: {43 - len(train_dist)}')
print(f'Classes with only 1 training example: {sum(1 for c in train_dist.values() if c == 1)}')
print(f'Classes with only 2 training examples: {sum(1 for c in train_dist.values() if c == 2)}')
print(f'Classes with <= 5 training examples: {sum(1 for c in train_dist.values() if c <= 5)}')

# Find classes in val but NOT in train
train_class_set = set(train_dist.keys())
val_only_classes = set(val_dist.keys()) - train_class_set

print(f'\n=== CRITICAL ISSUE ===')
print(f'Classes appearing in validation but NOT in training:')
if val_only_classes:
    dice_path = Path(r'C:\workspace\data\.cache\datasets\97ff5345-9d55-49af-9724-b6b1158569ae\annotations.json')
    with open(dice_path, 'r') as f:
        dice_data = json.load(f)

    categories = dice_data.get('categories', [])
    cat_id_to_name = {idx: cat['name'] for idx, cat in enumerate(categories)}

    for class_id in sorted(val_only_classes):
        count = val_dist[class_id]
        class_name = cat_id_to_name.get(class_id, 'UNKNOWN')
        print(f'  Class {class_id:2d} ({class_name:20s}): {count:2d} val objects, 0 train objects [ERROR]')

    print(f'\nTotal: {len(val_only_classes)} classes with NO training data!')
    print('This explains why validation metrics are 0 - the model has never seen these classes!')
else:
    print('  None - all val classes appear in training [OK]')
