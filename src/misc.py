import os
import math

import datasets
from matplotlib import pyplot as plt
import tqdm

def main():
    dataset_statistics()

def dataset_statistics():
    dataset = datasets.load_dataset(
        'laurievb/OpenLID-v2', token=os.environ.get("HUGGINGFACE_TOKEN"),
        features=datasets.Features({  # Present because without it, the function throws an exception
            'text': datasets.Value('string'),
            'language': datasets.Value('string'),
            'source': datasets.Value('string'),
            '__index_level_0__': datasets.Value('int64')
        })
    )

    lengths = []
    maximum = -math.inf
    bigger_than = {
        1024: 0,
        512: 0,
        256: 0,
    }
    step = 100
    for idx in tqdm.tqdm(range(0, len(dataset["train"]), step)):
        item = dataset["train"][idx]
        if item["text"]:
            lengths.append(len(item["text"]))
            maximum = max(maximum, len(item["text"]))
            for key in bigger_than.keys():
                bigger_than[key] += len(item["text"]) > key

    print(f"maximum: {maximum}")
    for key, val in bigger_than.items():
        print(f"Bigger than {key}: {val} ({(val * step) / len(dataset['train']) * 100} %)")

    if len(lengths) < 10_000_000:
        plt.hist(lengths, bins=[x for x in range(0, 2101, 100)])
        plt.show()


if __name__ == "__main__":
    main()