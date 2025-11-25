"""
Store all dataset/task relevant meta data here for passing them to the training script.
"""


def get_meta(task):
    if task == "LIDC":
        meta = {
            "description": "LIDC Lung Module Dataset (subset with 4 annotations)",
            "channels": 1,
            "all_data_path": "/root/",
            "masking_threshold": 0.5,
            "image_size": 128,
            "admissible_size": 128,
            "output_size": 128,
            "directory_name": "LIDC",
            "raters": 4,
            "num_filters": [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta
    if task == "isic3_style_concat":
        meta = {
            "description": "ISIC Skin Lesion Dataset with same split as style subsets ",
            "channels": 3,
            "all_data_path": "/root/autodl-tmp/isic256_3_style/isic256_3_style/",
            "masking_threshold": 0.5,
            "image_size": 256,
            "admissible_size": 340,
            "output_size": 252,
            "directory_name": "isic3",
            "raters": 3,
            "num_filters": [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta
    # Test comment2
