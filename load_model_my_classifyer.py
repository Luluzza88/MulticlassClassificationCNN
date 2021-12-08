#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st \nfrom PIL import Image\nimport numpy as np\nimport tensorflow as tf\nfrom keras.models import load_model\nfrom tensorflow.keras.preprocessing.image import load_img, img_to_array \n\n# st.set_option(\'deprecation.showfileUploaderEncoding\', False)\n# @st.cache(allow_output_mutation=True)\n\nst.write("""\n         # Waste Classification\n         """\n         )\n\nfile = st.file_uploader("Please upload an image", type=["jpg", "png"])\n\ndef make_predictions(image):\n  \n    my_classifier = load_model(\'my_classifier.h5\')\n\n    test_img = image.resize((150,150))\n    img_1 = img_to_array(test_img)/255.0\n    img1 = np.array([img_1])\n    predictions = my_classifier.predict(img1)\n    return predictions\n\nif file is None:\n  st.text(\'Please upload an image file\')\nelse:\n  image = Image.open(file)\n  st.image(image, use_column_width=True)\n  predictions = make_predictions(image)\n  class_names = [\'organic\', \'recycable\']\n  string="This image most likely displays: "+class_names[np.argmax(predictions)]\n  st.success(string)')


# In[2]:


#!pip install streamlit


# In[ ]:


get_ipython().system('streamlit run app.py')

