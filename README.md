# paintify

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact


###Project Steps###
Module 1: Image Preprocessing & Edge Detection (Person 1)
Responsibilities:
Load the image from the dataset or file input.
Convert the image to grayscale.
Apply edge detection (e.g., Canny edge detection).
Apply smoothing/denoising filters (Gaussian Blur or bilateral filter).
Color quantization using k-means clustering (optional).
Tasks:
Write functions to load and preprocess the image.
Implement edge detection.
Apply optional filters for noise reduction.
Deliverable: A Python module that returns an edge-detected and smoothed image.

Module 2: Color Quantization & Image Stylization (Person 2)
Responsibilities:
Reduce the number of colors in the image using color quantization techniques (e.g., k-means clustering or other algorithms).
Stylize the image by recoloring based on clusters.
Create a "cartoon effect" by combining quantized colors with edge detection results from Module 1.
Tasks:
Implement color quantization (e.g., k-means clustering).
Stylize the image by adding a cartoon-like effect.
Combine the quantized image with the edges.
Deliverable: A Python module that outputs a cartoonified image based on color quantization and edge detection.

Module 3: DeepAI API Integration for Additional Styles (Person 3)
Responsibilities:
Set up API integration with the DeepAI platform.
Send processed images to DeepAI to apply additional style transfer.
Handle API requests and responses.
Optionally, implement other advanced styles like pencil sketch, oil painting, etc.
Tasks:
Write functions to send images to the DeepAI API.
Fetch and display stylized images returned by the API.
Provide multiple style options for users to choose from (e.g., cartoon, sketch).
Deliverable: A Python module that interacts with DeepAI and returns stylized images.

Module 4: UI Development & Integration (Person 4)
Responsibilities:
Design a simple user interface (UI) for uploading images and displaying the cartoonified output.
Implement the UI using a framework like Flask (or Django if preferred).
Allow users to upload an image, apply the cartoonification, and display results.
Integrate the outputs from Module 1, Module 2, and Module 3 into the web interface.
Tasks:
Develop the front-end interface using HTML/CSS/JavaScript.
Create endpoints to handle image uploads and display the cartoonified result.
Test and ensure the UI works smoothly with the image processing modules.
Deliverable: A web application where users can upload images and view the cartoonified result in real-time.
