# Credit Card Default Classification with Gradient Boosted Trees

A gradient boosted trees classification model using TensorFlow is constructed to model and predict client credit card defaults, and is deployed on Google Cloud ML Engine. 

Data preprocessing is done with Apache Beam and data pipelines are created with DataFlow. Online prediction is then carried out with a REST API call to the model with desired input variables.

All the code is in `credit_card_Final` notebook, along with all the commands that are needed for building the data pipeline, creating the model, hyperparameter tuning, and deploying the final model to Google AI platform. All cautionary notes are also included. 

The data processing scripts can be found in `preprocess` folder, while the scripts for building the model are located in `gradient_boosted_model` folder. The data can also be found in the `data` subdirectory.
