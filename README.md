# PRODIGY_ML_01
**Task-02:-**  Create a K-means clustering algorithm to group customers of a retail store based on their purchase history. 

**Dataset:-** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

## **Libraries used:-**
* pandas: Data manipulation
* numpy: Numerical operations
* matplotlib.pyplot: Plotting on graph
* sklearn: K-means clustering of dataset

## **Importing Libraries:-**
* The code imports necessary libraries such as pandas for data manipulation, numpy for numerical operations, matplotlib.pyplot for plotting graphs, and sklearn.cluster.KMeans for performing K-means clustering.

## **Loading the Dataset:-**
*  The dataset Mall_Customers.csv is loaded into a pandas DataFrame named data.

## **Selecting Relevant Features:-**
*  Only the columns representing annual income and spending score are selected from the dataset and stored in the variable X.

## **Feature Scaling:-**
*  Since K-means clustering is sensitive to the scale of the features, the features in X are standardized using StandardScaler() from sklearn.preprocessing.

## **Elbow Method for Finding Optimal Number of Clusters:-**
* The Elbow Method is employed to determine the optimal number of clusters. This is done by fitting K-means clustering with a varying number of clusters (from 1 to 10) and calculating the Within-Cluster Sum of Squares (WCSS) for each number of clusters.
* WCSS is computed as the sum of squared distances between each point and the centroid of its assigned cluster.
* The WCSS values for different numbers of clusters are stored in the list wcss.

## **Plotting the Elbow Method Graph:-**
* The WCSS values are plotted against the number of clusters.
* The graph helps to visually identify the "elbow point", which indicates the optimal number of clusters. The elbow point is where the rate of decrease in WCSS slows down significantly.
* The plot is displayed using matplotlib.pyplot.

## **Applying K-means Clustering with Optimal Number of Clusters:-**
* Based on the Elbow Method analysis, the optimal number of clusters is determined to be around 5.
* K-means clustering is performed again with the optimal number of clusters using the KMeans class from sklearn.cluster.
* The fit_predict method is used to fit the model to the scaled data and obtain the cluster labels for each data point.

## **Visualizing the Clusters:-**
* A scatter plot is created to visualize the clusters.
* Each cluster is represented by a different color, and the centroids of the clusters are marked in yellow.
* The plot is displayed using matplotlib.pyplot.
