# Capsnet-with-scikeras-Tutorial
A colab notebook template to create a Multi-Input-Multi-Output Estimator using Sci-Keras and wrap it around a CapsNet Model to train the model using GridSearchCV hyperparmeter tuning. Read more about scikeras at: https://scikeras.readthedocs.io/en/latest/index.html

### The peculiar features of CapsNet that make the implementation non-trivial
- Capsule Layer - which is a non-conventional layer type, hence, user defined
- Training via Dynamic Routing - which again can not be trivially implemented.
- The model defining code, based on https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet.py, has a (X,y) -> (y,X) structure which provides a good opportunity to explore the capabilities that could be afforded by using Sci-Keras

### A few rooms for improvement (WIP)
- Figure out a way to make the code work in eager execution mode
- Figure out why multiple parallel jobs are not supported with GridSearchCV, i.e., make the estimator picklable.
- Figure out if it is possible to infer all shapes within the estimator from the data (no hardcoded shapes)
