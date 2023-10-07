# Training Image Classification Models with Hugging Face Transformers
This documentation provides a step-by-step guide to training an image classification model using the Hugging Face Transformers library. We will use the Vision Transformer (ViT) model and the NSFW-Filter dataset to identify image types in an image.

## 1. Dataset Preparation
Before we begin, we need to gather the dataset in the form of a .zip file.

For a simple example dataset, create two folders named "safe" and "no_safe." In the "safe" folder, include a collection of images that are considered safe and not explicit (non-NSFW). Additionally, add several images of various categories such as buildings, vehicles, road views, etc., to emphasize that these images are non-explicit and safe. Meanwhile, the "no_safe" folder should contain explicit NSFW images, specifically images without any clothes.
```
datasets.zip
  |-- no_safe
  | |-- image1.jpg
  | |-- image2.jpg
  | |-- ...
  | 
  |-- safe
  | |-- image1.jpg
  | |-- image2.jpg
  | |-- ...
```

## 2. Installation Prerequisites
Make sure you have the required libraries installed:
```
pip install transformers datasets
```

## 3. Login to HuggingFace Account
To share your models with the community, log in to your Hugging Face account:
```
from huggingface_hub import notebook_login
notebook_login()
```

## 4. Load NSFW-Filter Dataset
Load a small subset of the NSFW-Filter dataset for experimentation:
```
from datasets import load_dataset

link_datasets = "path/Dataset.zip"
dataset = load_dataset("imagefolder", data_files=link_datasets)
dataset = dataset.train_test_split(test_size=0.2)
```

## 5. Prepare Labels
Create a dictionary that maps labels to IDs and vice versa:
```
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
```

## 6. Data Preprocessing
Load the image processor for ViT and specify the transformation:
```
from transformers import AutoImageProcessor

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# Definition of transformations for data augmentation
transforms = ...

# Apply transformations to the dataset
dataset = dataset.with_transform(transforms)
```

## 7. Prepare Data for Training (PyTorch)
If you are using PyTorch, do the following steps:
```
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()
```

## 8. Prepare Data for Training (TensorFlow)
If you are using TensorFlow, do the following steps:
```
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors="tf")
```

## 9. Metric Evaluation
Load accuracy metrics for evaluation:
```
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
```

## 10. Training Configuration
Configure the training parameters, replacing the `output_dir` value with the desired output directory:
```
training_args = TrainingArguments(
    output_dir="./output",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)
```

## 11. Initialize Trainer and Model
Initialize ViT and Trainer models:
```
from transformers import AutoModelForImageClassification, Trainer

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
```

## 12. Model Training
Perform model training:
```
trainer.train()
trainer.push_to_hub()  # Upload the model to Hugging Face Hub
```

Congratulations, you have trained an image classification model using Hugging Face Transformers! Now you can use it for inference.



