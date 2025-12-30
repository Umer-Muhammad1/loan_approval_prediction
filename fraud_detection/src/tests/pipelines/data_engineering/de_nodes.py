
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
import json
import time
import great_expectations as gx
from great_expectations.datasource.fluent import PandasDatasource
import matplotlib.pyplot as plt
import seaborn as sns


def generate_correlation_heatmap_by_order(data: pd.DataFrame) -> None:
    
    # Compute correlation with 'loan_status'
    correlation_with_loan_status = data.corr(numeric_only=True)['loan_status'].sort_values(ascending=False)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the heatmap
    sns.heatmap(
        pd.DataFrame(correlation_with_loan_status),
        annot=True,
        cmap='coolwarm',
        fmt=".3f",
        ax=ax
    )

    # Set title
    ax.set_title('Correlation with Loan Status', fontsize=14)

    # Adjust layout
    fig.tight_layout()

    # Capture the figure
    img = plt.gcf()

    # Return the figure object
    return img


def plot_correlation_matrix(data):
    correlation_matrix = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        ax=ax,  
        cbar_kws={"shrink": 0.8}  
    )
    ax.set_title('Correlation Matrix Heatmap', fontsize=14)
    fig.tight_layout()
    img = plt.gcf()
    return img





    

def plot_outliers_all_columns(data):
 
   
    # Initialize the figure
    numerical_columns = data.select_dtypes(include=["number"]).columns
    n_features = len(numerical_columns)
    n_cols = 3  # Number of columns in the grid
    n_rows = 3  # Calculate rows needed

    # Create the grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, n_rows * 5))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Generate custom colors for each boxplot
    custom_colors = sns.color_palette("husl", len(numerical_columns))

    # Generate boxplots for each numerical column
    for i, col in enumerate(numerical_columns):
        sns.boxplot(
            x=data[col], 
            ax=axes[i], 
            color=custom_colors[i]  # Apply unique color to each boxplot
        )
        axes[i].set_title(f'Outliers for {col}', fontsize=12)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        # Create a figure with subplots
    
        
    
 # Set titles and labels
    
  
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img
            
        

def plot_loan_acceptance_by_categorical_features(data, categorical_features):
    
    # Set up the figure and axes
    n_features = len(categorical_features)
    n_cols = 2  # Number of columns
    n_rows = (n_features + 1) // n_cols  # Number of rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22,15))
    axes = axes.flatten()

    for i, feature in enumerate(categorical_features):
        # Calculate accepted and rejected percentages
        summary = data.groupby(feature).agg(
            accepted_loans=('loan_status', lambda x: (x == 1).sum()),
            total_loans_applied=('loan_status', 'size')
        )
        summary['accepted_percentage'] = (
            summary['accepted_loans'] / summary['total_loans_applied'] * 100
        )
        summary['rejected_percentage'] = 100 - summary['accepted_percentage']

        # Positioning for side-by-side bars
        categories = summary.index
        x = np.arange(len(categories))  # Label positions
        width = 0.35  # Width of bars

        # Current axis
        ax = axes[i]

        # Plotting side-by-side bars
        bar1 = ax.bar(
            x - width / 2,
            summary['accepted_percentage'],
            width,
            label='Accepted %',
            color='limegreen',
            edgecolor='black'
        )
        bar2 = ax.bar(
            x + width / 2,
            summary['rejected_percentage'],
            width,
            label='Rejected %',
            color='salmon',
            edgecolor='black'
        )

        # Adding percentage labels inside the bars
        for bar in bar1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='black')

        for bar in bar2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='black')

        # Formatting
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Percentage', fontsize=10)
        ax.set_title(f'Loan Acceptance vs Rejection by {feature}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.legend(fontsize=9)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img





def plot_distributions(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Age group distribution
    data['age_group'] = (data['person_age'] // 10) * 10
    sns.countplot(x='age_group', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Age Groups')

    # Income distribution
    income_bins = [0, 50000, 100000, 150000, 200000, float('inf')]
    income_labels = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K+']
    data['income_category'] = pd.cut(
        data['person_income'],
        bins=income_bins,
        labels=income_labels,
        right=False
    )
    sns.countplot(
        x='income_category',
        data=data,
        order=income_labels,
        ax=axes[0, 1]
    )

    # Employment length
    emp_length_bins = [-1, 0, 5, 10, 15, 20, float('inf')]
    emp_length_labels = [
        '<1 year', '1-5 years', '6-10 years',
        '11-15 years', '16-20 years', '20+ years'
    ]
    data['emp_length_category'] = pd.cut(
        data['person_emp_length'],
        bins=emp_length_bins,
        labels=emp_length_labels,
        right=False
    )
    sns.countplot(
        x='emp_length_category',
        data=data,
        order=emp_length_labels,
        ax=axes[0, 2]
    )
    axes[0, 2].set_xticklabels(emp_length_labels, rotation=45)

    # Loan amount
    loan_amnt_bins = [0, 5000, 10000, 15000, 20000, 25000, float('inf')]
    loan_amnt_labels = ['0-5K', '5K-10K', '10K-15K', '15K-20K', '20K-25K', '25K+']
    data['loan_amnt_category'] = pd.cut(
        data['loan_amnt'],
        bins=loan_amnt_bins,
        labels=loan_amnt_labels,
        right=False
    )
    sns.countplot(
        x='loan_amnt_category',
        data=data,
        order=loan_amnt_labels,
        ax=axes[0, 3]
    )

    # Histograms
    sns.histplot(data['cb_person_cred_hist_length'], bins=20, ax=axes[1, 0])
    sns.histplot(data['loan_int_rate'], bins=20, ax=axes[1, 1])
    sns.histplot(data['loan_percent_income'], bins=20, ax=axes[1, 2])

    axes[1, 3].axis('off')
    
    fig.tight_layout()
    
    # DO NOT close the figure - Kedro will handle that
    return fig



def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    df= data.drop_duplicates()
   
    return df

def plot_categorical_distributions(data: pd.DataFrame) -> None:
    
    # Select categorical columns
    categorical_cols = data.select_dtypes(include='object').columns

    # Number of categorical columns and layout settings
    num_plots = len(categorical_cols)
    rows = (num_plots + 3) // 4  # Calculate rows dynamically
    cols = min(num_plots, 4)  # Max columns per row is 4

    # Create a figure with adjusted size
    fig = plt.figure(figsize=(20,12 ))

    # Generate count plots for each categorical column
    for i, col in enumerate(categorical_cols):
        plt.subplot(rows, cols, i + 1)
        sns.countplot(x=data[col])
        plt.title(f'Distribution of {col}', fontsize=12)
        plt.xticks(rotation=45, ha='right')

    # Adjust layout and show the plots
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img


def plot_categorical_relations(
    df: pd.DataFrame, 
    categorical_features: list, 
    #continuous_feature: str, 
    hue_feature: str
) -> None:
    
    for cat_feature in categorical_features:
        fig= plt.figure(figsize=(12, 6))
        sns.swarmplot(
            data=df.sample(1000),  # Sample 1000 rows for the plot
            x=cat_feature,          # Categorical feature (on the x-axis)
            y=df['loan_amnt'],   # Continuous feature (on the y-axis)
            hue=hue_feature,        # Categorical feature (hue for color grouping)
            palette='viridis'       # Color palette
        )
        plt.title(f'Relation of Loan_amount and {cat_feature} by {hue_feature}')
        fig.tight_layout()
        img=plt.gcf()
    # Return the figure object
        return img
    
    
    
    
    
def plot_categorical_relations_grade(
    df: pd.DataFrame, 
    categorical_features: list, 
    #continuous_feature: str, 
    hue_feature: str
) -> None:
    
    for cat_feature in categorical_features:
        fig= plt.figure(figsize=(12, 6))
        sns.swarmplot(
            data=df.sample(1000),  # Sample 1000 rows for the plot
            x=df['loan_grade'],          # Categorical feature (on the x-axis)
            y=df['loan_amnt'],   # Continuous feature (on the y-axis)
            hue=hue_feature,        # Categorical feature (hue for color grouping)
            palette='viridis'       # Color palette
        )
        plt.title(f'Relation of Loan_amount and {cat_feature} by {hue_feature}')
        fig.tight_layout()
        img=plt.gcf()
    # Return the figure object
        return img
    
    
    



def plot_histograms_kde(
    df: pd.DataFrame, 
    hist_columns: list, 
    hue_column: str
) -> None:
    
    # Set up the grid layout
    num_plots = len(hist_columns)
    rows = (num_plots + 2) // 3  # Arrange in 3 columns
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()  # Flatten to iterate over all axes

    # Loop through the columns and create histograms
    for i, col in enumerate(hist_columns):
        sns.histplot(
            data=df,
            x=col,
            hue=hue_column,
            palette='viridis',
            kde=True,
            ax=axes[i]  # Specify the subplot axis
        )
        axes[i].set_title(f'Histogram of {col}', fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img

def validate_loan_data(df: pd.DataFrame):
    # 1. Initialize context
    context = gx.get_context()
    
    datasource_name = "loan_datasource"
    asset_name = "loan_asset"
    suite_name = "loan_suite"
    
    # 2. Get or add the pandas datasource
    try:
        ds = context.data_sources.get(datasource_name)
    except Exception:
        ds = context.data_sources.add_pandas(name=datasource_name)

    # 3. Add the Asset (without data initially)
    try:
        asset = ds.get_asset(asset_name)
    except Exception:
        asset = ds.add_dataframe_asset(name=asset_name)

    # 4. BUILD BATCH REQUEST (The Fix)
    # The error message says it needs 1 key called 'dataframe'
    batch_request = asset.build_batch_request(options={"dataframe": df})

    # 5. Handle the Suite
    try:
        suite = context.suites.get(name=suite_name)
    except Exception:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # 6. Get Validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    print("--- Running Great Expectations Validation ---")

    # 7. Add your Rules
    validator.expect_column_values_to_be_between("person_age", min_value=18, max_value=123)
    validator.expect_column_values_to_be_between("person_income", min_value=0)
    validator.expect_column_values_to_not_be_null("loan_status")
# --- Categorical Rules (Allowed Values) ---
    validator.expect_column_values_to_be_in_set(
        "person_home_ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    validator.expect_column_values_to_be_in_set(
        "loan_intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )
    validator.expect_column_values_to_be_in_set(
        "loan_grade", ["A", "B", "C", "D", "E", "F", "G"]
    )
    validator.expect_column_values_to_be_in_set(
        "cb_person_default_on_file", ["Y", "N"]
    )

    # --- Numerical Rules (Ranges) ---
    # Employment length shouldn't exceed age! But let's start with a basic range
    #validator.expect_column_values_to_be_between("person_emp_length", min_value=0, max_value=60)
    
    # Loan amount should be positive
    validator.expect_column_values_to_be_between("loan_amnt", min_value=500, max_value=50000)
    
    # Interest rate usually falls between 5% and 25%
    validator.expect_column_values_to_be_between("loan_int_rate", min_value=0, max_value=30)
    
    # Credit history length
    #validator.expect_column_values_to_be_between("cb_person_cred_hist_length", min_value=0, max_value=50)

    # --- Business Logic Rules ---
    # Loan percent of income should be a fraction between 0 and 1
    validator.expect_column_values_to_be_between("loan_percent_income", min_value=0, max_value=1.0)

    # 8. Run and Return
    results = validator.validate()
    # Visualization TODO
    #context.build_data_docs()
    #context.open_data_docs()
    
    if results["success"]:
            print("✅ Data Quality Validated!")
            return df
    else:
        print("❌ Data Quality Failed!")
        # ADD THIS LOOP TO SEE THE DETAILS:
        for result in results["results"]:
            if not result["success"]:
                # Use .expectation_config.expectation_type (as a property/attribute)
                # or result.expectation_config.type depending on your exact build
                rule_name = getattr(result.expectation_config, "expectation_type", "Unknown Rule")
                print(f"\nFAILED RULE: {rule_name}")
                
                # result.result is a dictionary containing the "partial_unexpected_list"
                unexpected = result.result.get("partial_unexpected_list", [])
                print(f"OFFENDING VALUES: {unexpected}")
        
        raise ValueError("Pipeline halted due to Data Quality issues.")

def quick_validate_csv(data: pd.DataFrame) -> pd.DataFrame:
    """Quick validation for a CSV file"""

    #df = pd.read_csv("data")
    return validate_loan_data(data)




def cast_loan_data_types(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Categorical casting (The big memory winners)
    cat_cols = [
        "person_home_ownership", "loan_intent", 
        "loan_grade", "cb_person_default_on_file"
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # 2. Downcasting Integers (The precision fix)
    df["person_age"] = df["person_age"].astype("int8")
    df["loan_status"] = df["loan_status"].astype("int8")
    df["id"] = df["id"].astype("int32")
    df["person_income"] = df["person_income"].astype("int32")
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].astype("int8")

    # 3. Downcasting Floats
    float_cols = ["person_emp_length", "loan_int_rate", "loan_percent_income"]
    for col in float_cols:
        df[col] = df[col].astype("float32")

    print("✅ Schema optimized for Docker/API deployment.")
    return df

#def validate_loan_data(data):
#    """Use in your Kedro pipeline"""
#    validator = SimpleLoanValidator()
#    results = validator.validate_data(data)
#    
#    # Optional: Fail pipeline if validation fails
#    if not results.success:
#        print("Warning: Data validation failed")
#        # Don't raise error, just log warning
#    
#    return data  # Pass the data through to the next node if valid


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    df= data.drop_duplicates()
   
    return df


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    
    data= data.drop(['id'], axis=1)
    data_1 = data[data['person_age'] < 90].reset_index(drop=True)
    data_2 = data_1[data_1['person_income'] < 1e6].reset_index(drop=True)
    df2 = data_2[data_2['person_emp_length'] < 60].reset_index(drop=True)

    return df2


def feature_target_split(data: pd.DataFrame):
    
    
    features = data.drop(columns='loan_status', axis =1)
    target = data['loan_status']
    return features, target
    
    

    
    
def train_test_df_split(features: pd.DataFrame, target: pd.Series):
    
    loan_amounts = features['loan_amnt'].copy()



 
    X_train, X_test, y_train, y_test , loan_amt_train, loan_amt_test= train_test_split(features, target,loan_amounts, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test , loan_amt_test


#def one_hot_encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
#    
#    # Concatenate X_train and X_test
#    combined_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
#    cat_cols= combined_data.select_dtypes(['object']).columns
#    # One-hot encode the categorical columns
#    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#    encoded_data = onehot_encoder.fit_transform(combined_data[cat_cols])
#    
#    # Convert encoded data to DataFrame
#    encoded_columns = onehot_encoder.get_feature_names_out(cat_cols)
#    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
#
#    # Drop the original categorical columns and add the encoded ones
#    combined_data_encoded = combined_data.drop(columns=cat_cols).join(encoded_df)
#    
#    # Split the data back into X_train and X_test
#    X_train_encoded = combined_data_encoded.iloc[:len(X_train), :]
#    X_test_encoded = combined_data_encoded.iloc[len(X_train):, :]
#
#    return X_train_encoded, X_test_encoded
#
#
#
#
#def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, stand_col: list):
#    
#    # Initialize the scaler
#    scaler = StandardScaler()
#
#    # Fit and transform the training data
#    X_train_scaled = scaler.fit_transform(X_train[stand_col])
#    X_test_scaled = scaler.transform(X_test[stand_col])
#
#    # Convert back to DataFrame
#    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=stand_col, index=X_train.index)
#    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=stand_col, index=X_test.index)
#
#    # Replace original columns with scaled columns
#    X_train_final = X_train.drop(columns=stand_col).join(X_train_scaled_df)
#    X_test_final = X_test.drop(columns=stand_col).join(X_test_scaled_df)
#
#    return X_train_final, X_test_final
import mlflow
def one_hot_encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # 1. Identify categorical columns
    cat_cols = X_train.select_dtypes(['object', 'category']).columns
    
    # 2. Initialize and Fit the encoder ONLY on X_train
    # 'handle_unknown="ignore"' is CRITICAL for FastAPI safety
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[cat_cols])
    
    # 3. Helper function to transform DataFrames
    def transform_data(df, enc, cols):
        encoded_array = enc.transform(df[cols])
        encoded_cols = enc.get_feature_names_out(cols)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        return df.drop(columns=cols).join(encoded_df)

    X_train_encoded = transform_data(X_train, encoder, cat_cols)
    X_test_encoded = transform_data(X_test, encoder, cat_cols)

    # Return the encoder so Kedro can save it for the API
    return X_train_encoded, X_test_encoded, encoder

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, stand_col: list):
    scaler = StandardScaler()
    
    # Fit ONLY on X_train
    scaler.fit(X_train[stand_col])

    # Transform both sets
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train[stand_col]), 
        columns=stand_col, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[stand_col]), 
        columns=stand_col, 
        index=X_test.index
    )

    X_train_final = X_train.drop(columns=stand_col).join(X_train_scaled)
    X_test_final = X_test.drop(columns=stand_col).join(X_test_scaled)
    X_test_final.to_csv("reference_data.csv", index=False)
    mlflow.log_artifact("reference_data.csv", artifact_path="monitoring")
    # Return the scaler for the API
    return X_train_final, X_test_final, scaler