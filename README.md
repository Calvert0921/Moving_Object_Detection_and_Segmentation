# Moving_Object_Detection_and_Segmentation

## File tree can be represented as:
```bash
project_ZhizhouFang/
├── data/
│   ├── CamVid/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── class_dict.csv      # Class-color mappings
│   └── annotations/      
│   │   ├── train_masks/    # Grayscale masks
│   │   ├── val_masks/
│   │   ├── test_masks/    
│   │   ├── gene_ann.py     # Annotation generator
│   │   ├── train_anns.json # COCO JSON files
│   │   ├── val_anns.json
│   │   └── test_anns.json 
├── src/
│   ├── data_aug.py         # Data Augmentation
│   ├── dataloader.py       # Data loading
│   ├── train.py            # Training script
│   ├── test.py             # Inference script
│   ├── test_seg.py         # Inference script only for segmentation
│   ├── find_class.py       # Count number of images with specific class
│   ├── model.py            # Model architecture
│   ├── evaluate.py         # Metrics calculation
│   ├── utils.py            # Helper functions
│   └── visualize.py        # Bounding box visualization
├── weights/                # Model weights
├── results/                # Predictions, plots
├── requirements.txt        # Conda / pip environment specs
└── README.md               # Setup / usage instructions
```

## To install all the dependencies:
### Run following command to create new environment
```bash
conda create --name object python=3.10
```
