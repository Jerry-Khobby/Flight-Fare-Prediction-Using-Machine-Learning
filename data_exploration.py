import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_config import get_logger

logger = get_logger(log_file="logs/kpi_exploration.log")

def perform_eda_kpis(df, save_plots=True, diagrams_folder="data/kpi-diagrams"):
    df = df.copy()
    os.makedirs(diagrams_folder, exist_ok=True)
    kpi_results = {}
    logger.info("Started EDA and KPI exploration")

    # Descriptive statistics
    airline_summary = df.groupby('Airline')['Total Fare'].describe()
    source_summary = df.groupby('Source')['Total Fare'].describe()
    destination_summary = df.groupby('Destination')['Total Fare'].describe()
    season_summary = df.groupby('Season')['Total Fare'].describe()
    
    kpi_results.update({
        'airline_summary': airline_summary,
        'source_summary': source_summary,
        'destination_summary': destination_summary,
        'season_summary': season_summary
    })
    
    # Log summaries (only top 5 rows to keep logs readable)
    logger.info("\nAirline Fare Summary (top 5):\n%s", airline_summary.head().to_string())
    logger.info("\nSource Fare Summary (top 5):\n%s", source_summary.head().to_string())
    logger.info("\nDestination Fare Summary (top 5):\n%s", destination_summary.head().to_string())
    logger.info("\nSeason Fare Summary (all):\n%s", season_summary.to_string())

    # Correlation matrix
    numerical_cols = ['Duration (hrs)', 'Base Fare', 'Tax & Surcharge', 'Total Fare', 'Days Before Departure']
    correlation_matrix = df[numerical_cols].corr(numeric_only=True)
    kpi_results['correlation_matrix'] = correlation_matrix
    logger.info("\nCorrelation matrix:\n%s", correlation_matrix.to_string())

    # Visualizations
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    if save_plots:
        path = os.path.join(diagrams_folder, "correlation_heatmap.png")
        plt.savefig(path)
        logger.info(f"Saved correlation heatmap to {path}")
    plt.close()
    
    plt.figure(figsize=(8,5))
    sns.histplot(df['Total Fare'], bins=50, kde=True)
    plt.title("Distribution of Total Fare")
    plt.xlabel("Total Fare")
    plt.ylabel("Frequency")
    if save_plots:
        path = os.path.join(diagrams_folder, "total_fare_distribution.png")
        plt.savefig(path)
        logger.info(f"Saved total fare distribution to {path}")
    plt.close()
    
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Airline', y='Total Fare', data=df)
    plt.xticks(rotation=45)
    plt.title("Fare Variation by Airline")
    if save_plots:
        path = os.path.join(diagrams_folder, "fare_by_airline.png")
        plt.savefig(path)
        logger.info(f"Saved boxplot of fare by airline to {path}")
    plt.close()
    
    monthly_avg = df.groupby('Departure Month')['Total Fare'].mean()
    monthly_avg.plot(kind='bar', figsize=(10,5))
    plt.title("Average Total Fare by Departure Month")
    plt.xlabel("Month")
    plt.ylabel("Average Total Fare")
    if save_plots:
        path = os.path.join(diagrams_folder, "avg_fare_by_month.png")
        plt.savefig(path)
        logger.info(f"Saved average fare by month plot to {path}")
    plt.close()

    # KPI Exploration
    avg_fare_airline = df.groupby("Airline")["Total Fare"].mean().sort_values(ascending=False)
    df['Route'] = df['Source'] + " -> " + df['Destination']
    popular_routes = df['Route'].value_counts().head(10)
    seasonal_avg_fare = df.groupby("Season")["Total Fare"].mean().sort_values(ascending=False)
    expensive_routes = df.groupby("Route")["Total Fare"].mean().sort_values(ascending=False).head(5)

    kpi_results.update({
        'avg_fare_airline': avg_fare_airline,
        'popular_routes': popular_routes,
        'seasonal_avg_fare': seasonal_avg_fare,
        'expensive_routes': expensive_routes
    })

    # Log KPI details
    logger.info("\nAverage fare per Airline:\n%s", avg_fare_airline.to_string())
    logger.info("\nTop 10 most popular routes:\n%s", popular_routes.to_string())
    logger.info("\nAverage fare per Season:\n%s", seasonal_avg_fare.to_string())
    logger.info("\nTop 5 most expensive routes:\n%s", expensive_routes.to_string())

    logger.info("EDA and KPI exploration complete")
    return kpi_results
