# Current working summary of project:
(Writeup/Masters Project Report will be included in the repo soon)
(Presentation will be added as well)

* Take 5437 objects from [Wittmann et al.](https://iopscience.iop.org/article/10.3847/1538-4365/ab4998) 
* 7 distinct classes from paper - grouped into 2 / binary classification
	* Member 1 (inside Perseus Cluster) or Background 0 (outside/behind PCC)
* Dark/faint objects make for a difficult training set - consider only bright objects, reduces training set
	* r_mag < 19.4
* Some hot pixel images / images where most pixels are red exist - thrown out
	* new training set is 272 images total. 
* Bolster training set by applying rotations for data augmentation
	* (all simple angles on unit circle - 30,45,60,90,120, and so on)
* Fed model grayscale images to train and classify - performance lowered so we are confident in the method including color. 
* Train/test split yields good results, but now we wish to move further out from the center of the PCC where Wittmann et al considered, so we use SDSS SQL queries to obtain more data.
* Use SQL to define selection regions in the Red Sequence and radially outward from the center of the cluster. We notice that stars appear here, even though their photometric flag suggests they are galaxies.
	* Currently trying to find if there are [flags](https://live-sdss4org-dr16.pantheonsite.io/algorithms/flags_detail/) that appear for stars and not galaxies but this is proving difficult
 	* One attempt was to look at the Spectroscopic redshift (z) of the object to discriminate between stars and galaxies' flags but their flags seem mixed
* Ignoring the presence of galaxies classified as stars, adding new data based off of Spectroscopic Redshift to the training set helped bolster model performance against independent new data.
* Confirmed that adding these new objects did in fact yield good results with an independent testset, searching 90 arcmins radially outward from the center of the Perseus Cluster, and subtracting out common objects with training data.

(8/27/24)
* Current steps are to ensure project is reproducable, will include requirements.txt and build steps. Restructuring to implement better practices and formats.
* Was recommended to explore k-fold cross validation for another check of model robustness

```
JPAstro/ (New name may be needed. Below are the most important files in each directory)
├── README.md
├── .gitignore
├── requirements.txt
├── Images/
│   ├── SDSS-png/ (initial 5437 images from Wittmann et al.)
│   ├── rotations-png/ (augmented data to train/test on)
│   │   ├── train/
│   │   ├── test/
│   └── WriteUpFigs/
├── Models/
├── Notebooks/
│   ├── perseusResNet50.ipynb (main notebook/driver)
│   ├── post-training.ipynb 
│   ├── Resnet Model Testing.ipynb
│   ├── flagChecking.ipynb
│   ├── Populate Dataset.ipynb
│   └── brightPairPlots.ipynb
├── Python/
│   ├── perseusResNet50.py
│   └── populateDataset.ipynb
└── SQL/
    ├── pcc_crossmatchQuery.txt
    └── radialSearchNoColor.txt
├── *data/
│   ├── raw/
│   │   ├── galaxy_images/
│   │   │   ├── cluster_1/
│   │   │   │   ├── galaxy_1.png
│   │   │   │   ├── galaxy_2.png
│   │   │   └── ...
│   │   ├── cluster_2/
│   │   └── ...
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── models/
│   ├── resnet50_pretrained.h5
│   ├── resnet50_finetuned.h5
│   └── model_architecture.py
├── results/
│   ├── training_logs/
│   │   ├── log_01.txt
│   │   └── log_02.txt
│   ├── model_predictions/
│   └── evaluation_metrics/
│       ├── confusion_matrix.png
│       ├── accuracy_report.txt
│       └── ...
└── config/
    ├── config.yaml
    └── hyperparameters.json
```
