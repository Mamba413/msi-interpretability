from ij import IJ 
import os
from ij import WindowManager as WM
import re

from ij.io import FileSaver

path1 = 'I:\\image_process_msi\\KR_data\\CRC_KR_TEST_MSIMUT\\MSIMUT'
save_path='I:\\image_process_msi\\KR_data\\KR_adjust\\KR_test_msi'

for filename in os.listdir(path1)[0:2]:
    if filename.endswith(".png"):
        imp = IJ.openImage(os.path.join(path1,filename))
        #这里就可以用插件啦
        IJ.run(imp,"Colour Deconvolution", "vectors=H&E")
        images = map(WM.getImage,WM.getIDList())
        for img in images:
           print('title is ',img.title)
           if(re.search(r'Colour\_1',img.title) is not None):
              fs = FileSaver(img)
              fs.saveAsPng(os.path.join(save_path,filename))
              print(img.title,'have saved')
           img.close()