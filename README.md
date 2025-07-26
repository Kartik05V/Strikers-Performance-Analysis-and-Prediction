# Football Striker Performance Analysis & Prediction

##  Project Overview
This project performs an end-to-end data analysis of football strikers' performance. The primary goal is to process raw data, uncover insights through statistical analysis, and build a machine learning model to classify players into different performance tiers ('Best' vs. 'Regular').

##  Project Workflow

1.  **Data Cleaning & Preprocessing:**
    * Loaded the dataset and handled missing values using `SimpleImputer` (median for numeric, most frequent for categorical).
    * Standardized data types, converting key metrics to integer format for analysis.

2.  **Exploratory Data Analysis (EDA):**
    * Performed descriptive analysis to get a statistical summary of the data.
    * Created visualizations using **Matplotlib** and **Seaborn** (pie charts, count plots) to analyze the distribution of players by nationality and footedness.

3.  **Statistical Analysis & Hypothesis Testing:**
    * Calculated key metrics like goal conversion rates.
    * Conducted hypothesis tests (**ANOVA**, **Correlation**) to determine if statistically significant differences or relationships exist between variables like nationality, hold-up play, and consistency.

4.  **Feature Engineering:**
    * Created a composite **'Total Contribution Score'** by summing multiple performance metrics.
    * Applied **Label Encoding** and **One-Hot Encoding** (`get_dummies`) to convert categorical features into a machine-readable format.

5.  **Unsupervised Learning - Clustering:**
    * Used the Elbow method to find the optimal number of clusters.
    * Applied **K-Means clustering** to segment players into distinct performance tiers ('Best strikers' vs. 'Regular strikers').

6.  **Supervised Learning - Classification:**
    * Built a **Logistic Regression** model to predict a striker's tier.
    * Scaled features using `StandardScaler` and split the data for training and testing.
    * Evaluated the model's performance using an accuracy score and a confusion matrix.

##  Tech Stack

* **Language:** Python
* **Core Libraries:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Statistical Analysis:** SciPy, Statsmodels
* **Machine Learning:** Scikit-learn

##  Key Findings & Insights

* **Player Profile:** The dataset showed that a majority of strikers are **right-footed**. The top-performing players consistently excelled in both scoring and creating chances (assists), highlighting the importance of a **well-rounded offensive skill set**.

* **Efficiency of Left-Footed Strikers:** While less common, left-footed players in this dataset exhibited a slightly higher **goal conversion rate**, suggesting a potential trend of more clinical finishing among this group.

* **Well-Rounded Contribution Defines "Best Strikers":** The K-Means clustering revealed that the **'Best Strikers'** cluster wasn't just defined by high goal counts. These players also scored significantly higher in **assists, aerial duels won, and big game performance**, indicating that elite strikers contribute to team play in multiple ways.

* **Hold-up Play and Consistency:** A moderate positive correlation was found between **'Hold-up Play'** and **'Big Game Performance'**. This suggests that a striker's ability to retain possession and bring teammates into the game is linked to their consistency in high-pressure situations.

* **Predictive Power of Composite Scores:** The engineered feature, **'Total Contribution Score'**, proved to be a powerful predictor in the Logistic Regression model. This confirms that a holistic view of a player's contributions is more effective for classification than relying on a single metric.

