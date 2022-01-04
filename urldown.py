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
  image = img.resize((80, 80))
  image_sequence = image.getdata()
  image_array = np.array(image_sequence)
  urllist.append(image_array)
