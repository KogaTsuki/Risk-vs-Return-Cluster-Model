import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def preprocess_data(df):

    # Create a copy of the dataframe with only needed columns
    # Beta represents market risk, 1Y Return represents historical performance
    data = df[['Symbol', 'Beta', '1Y Return %']].copy()
    
    # Remove rows with missing values to ensure clean data for clustering
    data = data.dropna()
    
    # Convert percentage to decimal for easier numerical processing
    # Example: 15.5% becomes 0.155
    data['1Y Return %'] = data['1Y Return %'] / 100
    
    # Exclude outliers using the IQR method
    for column in ['Beta', '1Y Return %']:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return data

def scale_features(data):

    # Create scaler object for standardizing features
    # StandardScaler transforms data to have mean=0 and variance=1
    scaler = StandardScaler()
    
    # Fit and transform the data
    # This step converts both Beta and Returns to comparable scales
    scaled_features = scaler.fit_transform(data[['Beta', '1Y Return %']])
    
    # Convert scaled numpy array back to dataframe for easier handling
    scaled_df = pd.DataFrame(scaled_features, columns=['Beta_scaled', 'Return_scaled'])
    scaled_df['Symbol'] = data['Symbol'].values
    
    return scaled_df, scaler

def perform_kmeans(data):

    # Initialize KMeans object
    # random_state ensures reproducibility of results
    kmeans = KMeans(n_clusters=4, random_state=42)
    
    # Fit the model and get cluster assignments
    # Each stock will be assigned to one of n_clusters groups
    clusters = kmeans.fit_predict(data[['Beta_scaled', 'Return_scaled']])
    
    return clusters


def analyze_clusters(data, original_data, clusters):

    # Add cluster assignments to the original data
    data_with_clusters = original_data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # Calculate comprehensive statistics for each cluster
    # This includes mean, min, max values and count of stocks
    cluster_stats = data_with_clusters.groupby('Cluster').agg({
        'Beta': ['mean', 'min', 'max', 'count'],
        '1Y Return %': ['mean', 'min', 'max']
    }).round(3)
    
    return cluster_stats

def plot_clusters(data, original_data, clusters, scaler, output_filename='clusters.png'):
    # Create a new figure with specified size
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with cluster assignments as color
    scatter = plt.scatter(original_data['Beta'], 
                          original_data['1Y Return %'],
                          c=clusters, 
                          cmap='viridis', 
                          edgecolor='k', 
                          s=100)
    
    # Add labels and title
    plt.xlabel('Beta (Risk)', fontsize=12)
    plt.ylabel('1-Year Return', fontsize=12)
    plt.title('Stock Clusters Based on Risk vs Return', fontsize=14)
    
    # Create legend with colors matching the scatter plot
    unique_clusters = np.unique(clusters)
    cmap = plt.cm.viridis
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster}', 
                   markerfacecolor=cmap(i / (len(unique_clusters) - 1)), markersize=10)
        for i, cluster in enumerate(unique_clusters)
    ]
    plt.legend(handles=handles, title='Cluster', loc='upper left')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save plot to file
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    
    # Read the data from CSV file
    df = pd.read_csv('../data/sp500_data_20241127.csv')
    
    # Preprocess the data to clean and prepare it
    processed_data = preprocess_data(df)
    print(f"Number of stocks after preprocessing: {len(processed_data)}")
    
    # Scale the features to normalize them
    scaled_data, scaler = scale_features(processed_data)
    
    # Perform clustering to group similar stocks
    clusters = perform_kmeans(scaled_data)
    
    # Create and save visualization
    plot_clusters(scaled_data, processed_data, clusters, scaler)
    print("\nCluster visualization has been saved as 'clusters.png'")
    
    # Print example stocks from each cluster for better understanding
    print("\nExample stocks from each cluster:")
    data_with_clusters = processed_data.copy()
    data_with_clusters['Cluster'] = clusters
    
    for cluster in range(4):
        cluster_stocks = data_with_clusters[data_with_clusters['Cluster'] == cluster]['Symbol'].head(5).tolist()
        print(f"\nCluster {cluster} stocks: {', '.join(cluster_stocks)}")
        
        # Calculate and print risk-return characteristics
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        avg_beta = cluster_data['Beta'].mean()
        avg_return = cluster_data['1Y Return %'].mean()
        print(f"Average Beta: {avg_beta:.2f}")
        print(f"Average Return: {avg_return:.2%}")