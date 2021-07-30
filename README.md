## Custom Detectron2 for TTA

> This is my custom Detectron2 for Test-Time Augmentation (TTA)

* dongbinna@postech.ac.kr

### How to curate the COCO Validation 2017 using [FiftyOne](https://voxel51.com/docs/fiftyone/integrations/coco.html)

* reference: [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html), [FiftyOne Dataset Object](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html)

<pre>

import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=1000,
    shuffle=True,
)

dataset.export(
    export_dir='./my_coco_val2017_1000/',
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    labels_path="./annotations/instances.json",
)

</pre>

### Mutiple Augmentations for TTA

* This implementation provides RandomBrightness, RandomContrast, RandomFlip, RandomSaturation, RandomRotation, ResizeShortestEdge at the same time.

<pre>

"""
[Your selection]
RandomFlip
RandomSaturation
ResizeShortestEdge
"""

my_dataset_mappter_tta = MyDatasetMapperTTA(
    # RandomBrightness, RandomContrast, RandomFlip, RandomSaturation, RandomRotation, ResizeShortestEdge
    [False, False, True, True, False, True]
)
tta_model = GeneralizedRCNNWithTTA(cfg, predictor.model, tta_mapper=my_dataset_mappter_tta)

</pre>
