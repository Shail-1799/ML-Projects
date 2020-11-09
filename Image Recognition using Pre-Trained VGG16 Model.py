#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

# Load Keras' VGG16 model that was pre-trained against the ImageNet database
model = vgg16.VGG16()


# In[113]:



def recog(i):
     
# Load the image file, resizing it to 224x224 pixels (required by this model)     
    img = image.load_img((i), target_size=(224, 224))

# Convert the image to a numpy array
    x = image.img_to_array(img)

# Add a fourth dimension (since Keras expects a list of images)
    x = np.expand_dims(x, axis=0)
    

# Normalize the input image's pixel values to the range used when training the neural network
    x = vgg16.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
    predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
    predicted_classes = vgg16.decode_predictions(predictions)

#     print("Top predictions for image:"+ str(i))

#     for imagenet_id, name, likelihood in predicted_classes[0]:
#          print("Prediction: {} - {:2f}".format(name, likelihood))
    
#     print( predicted_classes[0][0][1])

    return predicted_classes[0][0][1]

    print('\n')
    
    

    


# In[ ]:



pic = mpimg.imread("Path to the image to be recognised")
plt.figure()
plt.title("Recognition Result")
plt.imshow(pic)
       


       

