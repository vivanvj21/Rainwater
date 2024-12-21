import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from time import time

# Load data
data = pd.read_csv("./Datasets/Bengaluru Rainfall Data.csv")

# Preprocess the data
data.fillna(data.mean(), inplace=True)
data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)

# Calculate mean rainfall per month
months = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
mean_list = []
summed = 0.0
for i in months:
    mean_list.append(np.mean(data[i]))
    summed += mean_list[-1]

print(mean_list)
print(f'The predicted rainfall is: {summed} mm')
print(f'The expected rainfall collection for 1 year for a 30x40 site: {summed / 100 * 112} litres')

# Assuming data_dict is a dictionary loaded elsewhere
# Process salaries (assuming data_dict exists)
salaries = []
count = 0
for sample in data_dict.items():
    val = sample[1]['salary']
    if val != 'NaN':
        salaries.insert(count, val)
    count += 1
salaries.sort(reverse=True)

print(salaries[1], " ", salaries[len(salaries)-1])

# Data manipulation (assuming data_dict is defined and contains relevant data)
data_dict.pop("TOTAL", 0)

# Defining features for the model
feature_1 = "Temporal"
feature_2 = "Rain"
feature_3 = "total_investment"
poi = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

# Plot the scatter of features
for f1, f2, _ in finance_features:
    plt.scatter(f1, f2)
plt.ylabel("Rain")
plt.xlabel("Temporal")
plt.show()

# Train a KMeans clustering model
target, feature = targetFeatureSplit(data)
X_train, X_test, Y_train, Y_test = train_test_split(feature, target)
cluster = KMeans(n_clusters=3, max_iter=300, n_init=10)
t = time()
cluster.fit(X_train, Y_train)
print(f"Time to train: {round(time() - t, 3)} seconds")
pred = cluster.predict(X_test)

# Function to plot clusters
def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

# Call the Draw function to visualize clusters
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters_3ft.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named 'pred' found, no clusters to plot.")
