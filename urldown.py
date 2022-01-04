import urllib.request
from PIL import Image
import numpy as np
urllist = []
for i in train_url:
  try:
    urllib.request.urlretrieve(i.decode("utf-8") ,"gfg.jpg")
  except:
      pass
  img = Image.open("gfg.jpg")
  image_sequence = img.getdata()
  image_array = np.array(image_sequence)
  urllist.append(image_array)
