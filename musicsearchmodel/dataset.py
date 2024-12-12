#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import numpy as np
import random
import os

# Transformation and Preprocessing
class MapFunction():
    def __init__(self, imageSize):
        # define the image width and height
        self.imageSize = imageSize
    def decode_and_resize(self, imagePath):
        # read and decode the image path
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_png(image, channels=3)
        # convert the image data type from uint8 to float32 and then resize
        # the image to the set image size
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, self.imageSize)
        # return the image
        return image
    def __call__(self, anchor, positive, negative):
        anchor = self.decode_and_resize(anchor)
        positive = self.decode_and_resize(positive)
        negative = self.decode_and_resize(negative)
        # return the anchor, positive and negative processed images
        return (anchor, positive, negative)

class TripletGenerator:
    def __init__(self, datasetPath):
        # create an empty list which will contain the subdirectory
        # names of the `dataset` directory with more than one image
        # in it
        self.songNames = list()
        # iterate over the subdirectories in the dataset directory
        for folderName in os.listdir(datasetPath):
            # build the subdirectory name
            absoluteFolderName = os.path.join(datasetPath, folderName)
            # get the number of images in the subdirectory
            numImages = len(os.listdir(absoluteFolderName))
            # if the number of images in the current subdirectory
            # is more than one, append into the `songNames` list
            if numImages > 1:
                self.songNames.append(absoluteFolderName)
        # create a dictionary of song name to their image names
        self.allSongs = self.generate_all_songs_dict()
    def generate_all_songs_dict(self):
        # create an empty dictionary that will be populated with
        # directory names as keys and image names as values
        allSongs = dict()
        # iterate over all the directory names with more than one
        # image in it
        for songName in self.songNames:
            # get all the image names in the current directory
            imageNames = os.listdir(songName)
            # build the image paths and populate the dictionary
            spectrogramImages = [
                os.path.join(songName, imageName) for imageName in imageNames
            ]
            allSongs[songName] = spectrogramImages
        # return the dictionary
        return allSongs
       
    def get_next_element(self):
        # create an infinite generator
        while True:
            # draw a song at random which will be our anchor and
            # positive song
            anchorName = random.choice(self.songNames)
            # copy the list of song names and remove the anchor
            # from the list
            temporarySongs = self.songNames.copy()
            temporarySongs.remove(anchorName)
            # draw a song at random from the list of songs without
            # the anchor, which will act as our negative sample
            negativeSong = random.choice(temporarySongs)
            # draw two images from the anchor folder without replacement
            (anchorSong, positiveSong) = np.random.choice(
                a=self.allSongs[anchorName],
                size=2,
                replace=False
            )
            # draw an image from the negative folder
            negativeSong = random.choice(self.allSongs[negativeSong])
            # yield the anchor, positive and negative photos
            yield (anchorSong, positiveSong, negativeSong)





