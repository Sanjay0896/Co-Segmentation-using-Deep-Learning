Co-Segmentation-using-Deep-Learning 
Description

This repository contains the implementation of the research paper "Co-Detection in Images using Saliency and Siamese Networks" by Milan Zinzuvadiya, Vatsalkumar Dhameliya, Sanjay Vaghela, Sahil Patki, Nirali Nanavati, and Arnav Bhavsar. The paper addresses the co-detection problem in computer vision, which involves detecting common objects from multiple images using an integrated deep learning model.
Abstract

Co-Detection is an important problem in computer vision, involving the detection of common objects from multiple images. This project proposes an integrated deep learning model consisting of two networks for co-detection. The first network is a saliency network that generates saliency maps to detect objects in individual images. These maps are then passed as input to a Siamese neural network to determine whether the salient objects in both images are similar or different. The model achieves high-quality results on the iCoseg dataset.
Key Features

    Saliency Network: Utilizes the Non-Local Deep Features (NLDF) for Salient Object Detection to generate saliency maps.
    Siamese Network: Employs a Siamese neural network with triplet loss to ascertain the similarity between objects in different images.
    Deep Learning Integration: Combines convolutional neural networks (CNNs) and deep learning techniques to improve co-detection performance.
    Experimental Results: Achieves high-quality results on the iCoseg dataset, demonstrating effectiveness across various object classes with different backgrounds, poses, and contrasts.

Repository Contents

    data/: Contains the iCoseg dataset used for training and testing.
    models/: Includes the pre-trained saliency network and Siamese network models.
    notebooks/: Jupyter notebooks with detailed explanations and code for training and testing the networks.
    src/: Source code for the saliency network, Siamese network, and the overall co-detection model.
    results/: Contains visual results and performance metrics obtained from experiments.
    requirements.txt: List of dependencies and libraries required to run the project.

Getting Started
Prerequisites

    Python 3.x
    TensorFlow
    OpenCV
    NumPy
