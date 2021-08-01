## Custom Detectron2 for TTA

> This is my custom Detectron2 for Test-Time Augmentation (TTA)

* dongbinna@postech.ac.kr

### How to curate the COCO Validation 2017 using [FiftyOne](https://voxel51.com/docs/fiftyone/integrations/coco.html)

* References: [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html), [FiftyOne Dataset Object](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html)
* The following source code downloads 1,000 images of the COCO validation dataset.

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

* This implementation simply utilizes [RandomBrightness](https://github.com/facebookresearch/detectron2/blob/9eb0027d795bb9a38098bb05e2ceb273bfc9cf41/detectron2/data/transforms/augmentation_impl.py#L516), [RandomContrast](https://github.com/facebookresearch/detectron2/blob/9eb0027d795bb9a38098bb05e2ceb273bfc9cf41/detectron2/data/transforms/augmentation_impl.py#L490), [RandomFlip](https://github.com/facebookresearch/detectron2/blob/9eb0027d795bb9a38098bb05e2ceb273bfc9cf41/detectron2/data/transforms/augmentation_impl.py#L76), [RandomSaturation](https://github.com/facebookresearch/detectron2/blob/9eb0027d795bb9a38098bb05e2ceb273bfc9cf41/detectron2/data/transforms/augmentation_impl.py#L542), [RandomRotation](https://github.com/facebookresearch/detectron2/blob/9eb0027d795bb9a38098bb05e2ceb273bfc9cf41/detectron2/data/transforms/augmentation_impl.py#L231), [ResizeShortestEdge](https://github.com/facebookresearch/detectron2/blob/9eb0027d795bb9a38098bb05e2ceb273bfc9cf41/detectron2/data/transforms/augmentation_impl.py#L128) at the same time.
* Used codes are based on the Detectron2 [transforms](https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/transforms/augmentation_impl.py).

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
