# ML_Algorithm_Analysis
## Repo Break Down
1. data/ : a folder to store data
    a. raw/ : a folder to store raw data
    b. processed/ : a folder to store processed data
2. src/ : a folder to store the source code for each algorithm.
    data_loader.py: contain the script that load and process the data
    Each algorithm is a folder that have: 
    a. evaluate.py: script to evaulate the model
    b. model.py: script to define the model
    c. predict.py: script to run the model in inference time
    d. train.py: script to train the model
    e. utils.py: contains helper functions
3. results/ : a folder that store the result
    a. models/ : contains the trained model
    b. results/ : contains the results