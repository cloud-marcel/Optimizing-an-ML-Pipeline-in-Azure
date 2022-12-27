# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The open [bankmarketing dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) includes information about bank clients and how they response to a marketing campaign via phone calls. These responses can be either 'Yes' or 'No'. That means we have a binary classification problem here in order to predict if the client will subscribe a term deposit. 
The dataset includes 20 features (like age or job) of 32.950 clients and the class label ('yes' or 'no') in the last column.

The highest accuracy was achieved by an automatic generated model with the `VotingEnsemble` algorithm via AutomML with a metric of **91.75%**. In comparison, the hyperparameter tuning using Hyperdrive with an underlying `LogisticRegression` model reached an accuracy of **90.91%**.

## Scikit-learn Pipeline

The architecture contains of two elements: 
- The Pyhton scripts (`train.py`) includes the `Scikit-learn` model, functions to  read the dataset from the URL and saving it as `TabularDatasetFactory` object, cleaning the dataset, splitting it into training and test data und receiving the final metric score.
- The Jupyter notebook is hosted on a compute instance and automates the steps in order to tune the hyperparameters of the `LogisticRegression` model. To do so, it creates the compute clusters, specifies the configuration of the hyperparameter tuning engine and runs the experiment.

### Parameter Sampler
In the notebook, a parameter sampler of the Class [RandomParameterSampling](https://learn.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) was chosen which supports discrete and continuous hyperparameters. It was specified as follows:
```
ps = RandomParameterSampling( {
    "--C" : uniform(0.1,1),
    "--max_iter" : choice(50, 100, 150, 200)
    }
)
```
The [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) says about the parameters:
- `C` (float): Inverse of regularization strength; must be a positive float. That means, we need a continuous variable here, sampled via the `uniform` method.
- `max_iter` (int): Maximum number of iterations taken for the solvers to converge. For this, I used a discrete variable with the `choice` function.
With limiting the range of possible parameters, I was able to shrink the training resources and time.

### Early Stopping Policy
The [BanditPolicy](https://learn.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py) defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor of the evaluation metric with respect to the best performing run will be terminated.
The policy was implemented in the notebook as you can see below:
```
policy = BanditPolicy(
    evaluation_interval=1, 
    slack_factor=0.1,
    delay_evaluation=4
)
```
The policy automatically terminates poorly performing runs and improves computational efficiency so I prevented my training procedure to deal with not promising hyperparameters.

## AutoML
AutoML is a new feature of the Azure Cloud to automate the time consuming, iterative tasks of machine learning model development. In contrast to Hyperparameter tuning with HyperDrive, you don't need a model which is specified by the ML engineer before the training. Rather AutoML finds a model by using different algorithms and parameters trying to improve the specified metrics.

The orchestration is done in the same Notebook, but you do not need a training script here. That is why I received the data from the URL above. For splitting the dataset into training and test data with the same ration as above, I used the `random_split` method of `TabularDatasetFactory` class. 
The AutoML engine only works with a cleaned training dataset. For this reason, I used the `clean_data` method from the `train.py` script.

The implemented AutoML Config looks as follows:
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=25,
    task='classification',
    primary_metric='accuracy',
    training_data=train_data_cleaned,
    label_column_name='y',
    n_cross_validations=3
)
```
With the `experiment_timeout_minutes` parameter the procedure of finding an optimal model is limited. Here, 25 minutes were enough to beat the model tuned by HyperDrive.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
