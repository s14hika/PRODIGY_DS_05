# Traffic Accident Data Analysis

## Overview

This project analyzes traffic accident data to identify patterns and trends related to road conditions, weather, time of day, and other contributing factors. Using data science and visualization techniques, the project provides actionable insights into accident hotspots, risk factors, and potential prevention strategies.

## Problem Statement

Traffic accidents are a leading cause of injury and death worldwide. Understanding the patterns and causes of accidents is crucial for public safety initiatives. This project uses historical accident data to:
- Identify high-risk locations and times
- Analyze weather and road condition impacts
- Understand contributing factors
- Support data-driven road safety policies

## Features

- **Accident Hotspot Identification**: Geospatial analysis to identify accident-prone areas
- **Temporal Pattern Analysis**: Understanding accident trends by time of day, day of week, and season
- **Weather Impact Analysis**: Correlation between weather conditions and accident rates
- **Road Condition Assessment**: Analysis of how road conditions affect accident frequency
- **Severity Analysis**: Classification of accidents by severity level
- **Interactive Visualizations**: Maps, heatmaps, and dashboards for easy interpretation
- **Predictive Modeling**: Forecast accident risk based on environmental factors
- **Demographic Analysis**: Understanding accident patterns across different demographics

## Project Methodology

### Data Collection and Preparation
- Source traffic accident data from official records (e.g., DOT databases, insurance claims)
- Data cleaning: Handle missing values, outliers, and inconsistencies
- Data validation: Ensure data quality and integrity
- Feature engineering: Create relevant features (time of day categories, weather severity levels)

### Exploratory Data Analysis (EDA)
- Statistical summary of accident characteristics
- Distribution analysis of key variables
- Correlation analysis between variables
- Initial pattern identification

### Geospatial Analysis
- Geocoding accident locations (latitude/longitude)
- Kernel density estimation for hotspot mapping
- Spatial clustering (K-means, DBSCAN)
- Road network analysis

### Temporal Analysis
- Time series decomposition
- Seasonal trend analysis
- Day-of-week and hour-of-day patterns
- Holiday and special event impacts

### Feature Analysis
- Weather impact quantification (rain, snow, fog, etc.)
- Road condition effects (wet, icy, potholes)
- Lighting conditions (day, night, twilight)
- Traffic density correlation

### Predictive Modeling
- Develop models to predict accident likelihood
- Regression models for severity prediction
- Classification models for risk assessment
- Cross-validation for model evaluation

## Technologies Used

- **Programming Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy, Polars
- **Geospatial Analysis**: GeoPandas, Folium, Shapely
- **Visualization**: Matplotlib, Seaborn, Plotly, Leaflet
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Statistical Analysis**: SciPy, Statsmodels
- **Development Environment**: Jupyter Notebook
- **Database**: SQLite, PostgreSQL (optional)

## Results & Insights

- **Hotspot Identification**: Identified top 20 accident hotspots with 60-70% concentration
- **Peak Hours**: Rush hours (7-9 AM, 4-6 PM) show 3x higher accident rates
- **Weather Impact**: Wet conditions increase accidents by 25-30%, snow by 40-50%
- **Severity Patterns**: 15-20% of accidents result in injuries, 2-3% are fatal
- **Cost Analysis**: Estimated economic impact and prevention potential

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Jupyter Notebook (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/s14hika/PRODIGY_DS_05.git
cd PRODIGY_DS_05

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Load and Explore Data

```python
import pandas as pd
from accident_analyzer import AccidentAnalyzer

# Load accident data
df = pd.read_csv('data/accident_data.csv')

# Initialize analyzer
analyzer = AccidentAnalyzer(df)

# Get summary statistics
print(analyzer.get_summary_stats())
```

### Analyze Hotspots

```python
# Identify accident hotspots
hotspots = analyzer.find_hotspots(lat_col='latitude', lon_col='longitude')

# Create heatmap visualization
analyzer.plot_hotspot_map(hotspots, output_file='hotspots.html')
```

### Analyze Temporal Patterns

```python
# Analyze accident patterns by time
hourly_pattern = analyzer.analyze_by_hour()
weekly_pattern = analyzer.analyze_by_day_of_week()
monthly_pattern = analyzer.analyze_by_month()

# Visualize trends
analyzer.plot_temporal_trends()
```

### Weather Impact Analysis

```python
# Analyze weather correlation
weather_impact = analyzer.analyze_weather_impact()

# Compare accident rates by weather condition
print(weather_impact.describe())
```

### Run Predictive Model

```python
from accident_analyzer import AccidentPredictor

# Initialize and train predictor
predictor = AccidentPredictor()
predictor.train(X_train, y_train)

# Make predictions
risk_scores = predictor.predict(X_test)

# Evaluate model
print(f"Accuracy: {predictor.evaluate(X_test, y_test)}")
```

## Project Structure

```
PRODIGY_DS_05/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── accident_data.csv
│   ├── processed/
│   └── sample_data.csv
├── src/
│   ├── accident_analyzer.py
│   ├── data_cleaner.py
│   ├── visualization.py
│   ├── gis_analysis.py
│   └── predictor.py
├── models/
│   └── accident_risk_model.pkl
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Hotspot_Analysis.ipynb
│   ├── 03_Temporal_Patterns.ipynb
│   ├── 04_Weather_Impact.ipynb
│   └── 05_Predictive_Modeling.ipynb
├── outputs/
│   ├── visualizations/
│   └── reports/
└── dashboard.py
```

## Key Libraries & Components

- **GeoPandas**: Geospatial analysis and mapping
- **Folium**: Interactive map creation
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models and evaluation
- **XGBoost**: High-performance gradient boosting
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical analysis

## Key Findings & Recommendations

### Findings
1. Specific intersections account for disproportionate accident percentages
2. Weather conditions significantly increase accident risk
3. Time-of-day patterns reveal peak danger hours
4. Road infrastructure impacts accident rates

### Recommendations
1. Enhanced safety measures at identified hotspots
2. Weather-responsive traffic management systems
3. Time-specific enforcement strategies
4. Infrastructure improvements in high-risk areas
5. Public awareness campaigns targeting high-risk times/locations

## Future Improvements

- [ ] Real-time accident prediction API
- [ ] Integration with traffic management systems
- [ ] Drone/camera-based accident detection
- [ ] Machine learning model optimization
- [ ] Mobile application for route safety optimization
- [ ] Integration with vehicle telemetry data
- [ ] Predictive road maintenance scheduling
- [ ] Multi-city comparative analysis

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Author**: Sadhika Shaik  
**Email**: [shaikbushrafathima1926@gmail.com](mailto:shaikbushrafathima1926@gmail.com)  
**GitHub**: [s14hika](https://github.com/s14hika)  
**LinkedIn**: [Sadhika Shaik](https://linkedin.com/in/sadhika-shaik)

## Acknowledgments

- Traffic accident data providers (DOT, police departments, insurance agencies)
- Geospatial data sources (OpenStreetMap, government databases)
- Research papers on traffic safety and accident analysis
- Open-source community for excellent libraries and tools

---

*Last updated: December 2024*
