def name = getProjectEntry().getImageName() + '.txt'
setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049 ", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581 ", "Background" : " 255 255 255 "}');
createSelectAllObject(true);
runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "backgroundRadius": 15.0,  "medianRadius": 0.0,  "sigma": 3.0,  "minArea": 10.0,  "maxArea": 1000.0,  "threshold": 0.1,  "maxBackground": 2.0,  "watershedPostProcess": true,  "cellExpansion": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true,  "thresholdCompartment": "Nucleus: Eosin OD mean",  "thresholdPositive1": 0.2,  "thresholdPositive2": 0.4,  "thresholdPositive3": 0.6000000000000001,  "singleThreshold": true}');
selectDetections();
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"downsample": 1.0,  "region": "ROI",  "tileSizePixels": 200.0,  "colorOD": false,  "colorStain1": true,  "colorStain2": true,  "colorStain3": false,  "colorRed": false,  "colorGreen": false,  "colorBlue": false,  "colorHue": false,  "colorSaturation": false,  "colorBrightness": false,  "doMean": false,  "doStdDev": false,  "doMinMax": false,  "doMedian": false,  "doHaralick": true,  "haralickDistance": 1,  "haralickBins": 32}');
path1 = '/G:/知识汇总/计算机软件与语言/生物图像处理软件/QuPath/示范例子/'
mkdirs(path1)
path = buildFilePath(path1, name)
saveDetectionMeasurements(path)
print 'Results exported to ' + path