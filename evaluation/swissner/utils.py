from datasets import load_dataset, concatenate_datasets


def custom_load_dataset(path: str, *args, **kwargs):
    if path.startswith("swissner"):
        dataset = load_dataset("ZurichNLP/swissner")
        language = path.split("_")[-1]
        dataset["test"] = dataset[f"test_{language}"]
        dataset["train"] = dataset[f"test_{language}"]
        dataset["validation"] = dataset[f"test_{language}"]
        return dataset
    elif path == "wikineural_defrit":
        dataset = load_dataset("Babelscape/wikineural")
        languages = ["de", "fr", "it"]
        dataset["train"] = concatenate_datasets([dataset[f"train_{lang}"] for lang in languages])
        dataset["validation"] = concatenate_datasets([dataset[f"val_{lang}"] for lang in languages])
        dataset["test"] = concatenate_datasets([dataset[f"test_{lang}"] for lang in languages])
        tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        id_to_tag = {v: k for k, v in tags.items()}
        dataset = dataset.map(
            lambda x: {"ner_tags": [id_to_tag[t] for t in x["ner_tags"]]}
        )
        return dataset
    elif path == "wikineural":
        dataset = load_dataset("Babelscape/wikineural")
        languages = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ru"]
        dataset["train"] = concatenate_datasets([dataset[f"train_{lang}"] for lang in languages])
        dataset["validation"] = concatenate_datasets([dataset[f"val_{lang}"] for lang in languages])
        dataset["test"] = concatenate_datasets([dataset[f"test_{lang}"] for lang in languages])
        tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        id_to_tag = {v: k for k, v in tags.items()}
        dataset = dataset.map(
            lambda x: {"ner_tags": [id_to_tag[t] for t in x["ner_tags"]]}
        )
        return dataset
    else:
        return load_dataset(path, *args, **kwargs)
