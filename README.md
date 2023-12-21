# Current working summary of project:

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


