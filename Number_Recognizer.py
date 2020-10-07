#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
# load weights into new model
cnn.load_weights("model.h5")
print("Loaded model from disk")


# In[2]:


cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[3]:


##pip install pygame


# In[4]:


import pygame, sys
from pygame.locals import *
import ctypes


# In[8]:


def main():
    pygame.init()
    BLACK = (255, 255, 255)
    WHITE = (0, 0, 0)

    mouse_position = (0, 0)
    drawing = False
    screen = pygame.display.set_mode((280,280))
    screen.fill(WHITE)
    pygame.display.set_caption("Number Pad")
    fname="number.jpg"
    
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEMOTION:
                if (drawing):
                    mouse_position = pygame.mouse.get_pos()
                    pygame.draw.circle(screen, BLACK, mouse_position,10, 10)
            elif event.type == MOUSEBUTTONUP:
                mouse_position = (0, 0)
                drawing = False
            elif event.type == MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == KEYUP:
                if event.key == pygame.K_BACKSPACE:
                    screen.fill(WHITE)
                elif event.key == pygame.K_RETURN:
                    pygame.image.save(screen,fname)
                    test_image = image.load_img('number.jpg',color_mode='grayscale', target_size = (28, 28))
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis = 0)
                    result = cnn.predict(test_image)
                    num = result.argmax()
                    ctypes.windll.user32.MessageBoxW(0,"The machine guessed: "+str(num), "Number Recognition",0)

        pygame.display.update()

if __name__ == "__main__":
    main()

