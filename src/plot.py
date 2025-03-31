from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# fetch dataset 
adult = fetch_ucirepo(id=2) 

# data (as pandas dataframes) 
x = adult.data.features 
y = adult.data.targets
y.loc[:, "income"] = y["income"].replace({"<=50K.": "<=50K", ">50K.": ">50K"})
y.loc[:, "income"] = y["income"].replace({"<=50K": "<=50K", ">50K": ">50K"})


# Create a new figure for each column
for col in x.columns:
    plt.figure(figsize=(6, 5))
    
    if x[col].dtype == 'object':  # Categorical column
        # Group data by income and count the occurrences for each category
        cat_counts = pd.crosstab(x[col], y['income'])

        # Plot count vs category bar chart for both income labels
        cat_counts.plot(kind='bar', stacked=False, color=['skyblue', 'orange'], width=0.8)
        
        plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    else:  # Continuous numerical column
        # Scatter plot with different colors for each label
        scatter = plt.scatter(x[col], np.zeros_like(x[col]), c=y['income'].map({'<=50K': 'blue', '>50K': 'red'}))

        # Set title and labels
        plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Value (y=0)')
        plt.yticks([])  # Hide y-axis ticks since all points are at y=0
        
        # Create custom legend handles
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='<=50K')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='>50K')

        # Add custom legend to the plot
        plt.legend(handles=[blue_patch, red_patch])

    plt.tight_layout()
    plt.show()  # Display the plot for the current column and wait for the user to close it before moving to the next one
sys.exit()