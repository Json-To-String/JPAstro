# # Generate Rotation data

def applyRotations(originalDf, outDir):

	# files and labels as numpy arrays
	files = originalDf['files'].to_numpy()
	label = originalDf['labels'].to_numpy()
	#rotDir = 'rotations'
	rotDir = 'rotations-png'
	originalDir = 'SDSS1/'
	# rotDir = 'rotations256'

	rotFilenames = list()
	rotLabels = list()

	#angle = [90, 180, 270, 360]
	angle = [30, 45, 60, 90, 120, 135, 150, 180, 210, 235, 240, 270, 300, 315, 330, 360]

	# Use PIL to rotate image on angles in list
	for ang in angle:
		for f, l in zip(files, label):
			imgString = originalDir + f
			im = PIL.Image.open(imgString)
			out = im.rotate(ang)
			# generated filename
			outString = f'{rotDir}/{outDir}/{f[:-5]}_rot{ang}_label={l}.png'
			# filename relative to working directory
			dfString = f'{outDir}/{f[:-5]}_rot{ang}_label={l}.png'

			out.save(outString)
			rotFilenames.append(dfString)
			rotLabels.append(l)

			rotationDf = pd.DataFrame({'files': rotFilenames,
						'labels': rotLabels})

	return(rotationDf)
