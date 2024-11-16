TrashCAN 1.0 

Jungseok Hong, Michael Fulton, Junaed Sattar 
Related Paper: https://arxiv.org/abs/2007.08097

IF USING CITE THE ABOVE PAPER.


For questions, issues, etc, contact: irvlab@umn.edu
----------------------------------------------------------------

This folder contains the actual data of the TrashCAN dataset:

	- instance_version/ contains the images and annotations for the instance-labeled version of TrashCAN
	- material_version/ contains the images and annotations for the material-labeled version of TrashCAN
	- original_data/ contains the original images and annotations for TrashCAN, prior to cleaning and transformation to COCO format.
	- scripts contains data processing scripts, in this case only one script, trash_can_coco.py.
		To use it, run the following command: python trash_can_coco.py [material/instance]
		
		You need to run this in the same folder as the annotations/ and images/ directories in the original_data folder.
