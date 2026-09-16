from ner_datasets import load_ner_dataset_from_conll
from ner_datasets.sampling import sample_few_shot

data = load_ner_dataset_from_conll("/share/nverdha/data/ler/validation.conllu")
train, eval_, counter, classes, *_ = sample_few_shot(
    data.samples, classes=["ORG"], negative_classes=["NRM"], num_negative_examples=10
)
print(f"counter_examples: {len(counter)}")
for c in counter:
    print(c)
