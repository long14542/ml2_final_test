import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from config import Config


class DataUtils:
    def __init__(self, raw_data):
        self.raw_data = raw_data.copy()
        # Normalize column names: remove spaces and special characters
        self.raw_data.columns = [self._clean_column_name(col) for col in self.raw_data.columns]
        self.processed_data = None
        self.numerical_features = []
        self.categorical_features = []
        self.target_column = self._find_target_column()
        self.plots = {}
        self.scaler = RobustScaler()  # Changed from StandardScaler to RobustScaler

    def _clean_column_name(self, column_name):
        """Clean column names by removing spaces and converting to lowercase"""
        return re.sub(r'[^a-zA-Z0-9_]', '', column_name.strip().lower())

    def _find_target_column(self):
        """Find the target column, such as 'loan_status' or similar"""
        target_candidates = ['loan_status', 'loanstatus', 'status', 'approved']
        for col in self.raw_data.columns:
            if col.lower() in target_candidates:
                return col
        return None

    def analyze_dataset(self):
        """Analyze data and create statistical plots"""
        if self.target_column:
            print(f"   Target column detected: {self.target_column}")

            # Classify features
            for col in self.raw_data.columns:
                if col == self.target_column or col == 'loan_id':  # Ignore ID and target columns
                    continue

                if self.raw_data[col].dtype == 'object' or self.raw_data[col].nunique() < 10:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)

            # Create analysis plots
            self._create_distribution_plots()
            self._create_correlation_matrix()
            self._create_target_analysis_plots()
        else:
            print("   No suitable target column found.")

    def _create_distribution_plots(self):
        """Create distribution plots for numerical features"""
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(self.numerical_features[:6]):  # Show up to 6 features
            plt.subplot(2, 3, i + 1)
            sns.histplot(self.raw_data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()

        self.plots['distributions'] = plt.gcf()
        plt.close()

    def _create_correlation_matrix(self):
        """Create a correlation matrix for numerical features"""
        corr_matrix = self.raw_data[self.numerical_features].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation matrix between numerical features')
        plt.tight_layout()

        self.plots['correlation'] = plt.gcf()
        plt.close()

    def _create_target_analysis_plots(self):
        """Create plots analyzing relationships between features and the target variable"""
        # Count plot for the target variable
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.target_column, data=self.raw_data)
        plt.title(f'Target variable distribution - {self.target_column}')
        plt.tight_layout()
        self.plots['target_distribution'] = plt.gcf()
        plt.close()

        # Boxplots for the top 4 important numerical features
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(self.numerical_features[:4]):
            plt.subplot(2, 2, i + 1)
            sns.boxplot(x=self.target_column, y=feature, data=self.raw_data)
            plt.title(f'{feature} by {self.target_column}')
        plt.tight_layout()
        self.plots['feature_target_relations'] = plt.gcf()
        plt.close()

    def save_plots(self, output_dir=None):
        """Save plots to the output directory"""
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR

        os.makedirs(output_dir, exist_ok=True)

        for name, fig in self.plots.items():
            fig.savefig(os.path.join(output_dir, f'{name}.png'))

    def process_data(self):
        """Process and clean data"""
        df = self.raw_data.copy()

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Drop ID column if present
        if 'loan_id' in df.columns:
            df.drop('loan_id', axis=1, inplace=True)

        # Handle missing values
        # For numerical features: fill with median
        for feature in self.numerical_features:
            if df[feature].isnull().sum() > 0:
                df[feature].fillna(df[feature].median(), inplace=True)

        # For categorical features: fill with the most common value
        for feature in self.categorical_features:
            if df[feature].isnull().sum() > 0:
                df[feature].fillna(df[feature].mode()[0], inplace=True)

        # Encode categorical features
        for feature in self.categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])

        # Encode the target variable if needed
        if df[self.target_column].dtype == 'object':
            le = LabelEncoder()
            df[self.target_column] = le.fit_transform(df[self.target_column])

        # Normalize numerical features
        num_features = [f for f in self.numerical_features if f in df.columns]
        if num_features:
            df[num_features] = self.scaler.fit_transform(df[num_features])

        # Create new features
        self._create_new_features(df)

        self.processed_data = df
        return df

    def _create_new_features(self, df):
        """Create 5 new features as required"""
        # 1. TOTAL_ASSETS
        df['total_assets'] = df['residential_assets_value'] + \
                             df['commercial_assets_value'] + \
                             df['luxury_assets_value'] + \
                             df['bank_asset_value']

        # 2. DEBT_RECOVERY_RATIO
        df['debt_recovery_ratio'] = (df['residential_assets_value'] + df['commercial_assets_value']) / \
                                    (df['loan_amount'] + 1e-6)

        # 3. REPAYMENT_CAPACITY_INDEX
        df['repayment_capacity'] = (df['income_annum'] * 0.5 + df['total_assets'] * 0.3) / \
                                   (df['loan_amount'] / df['loan_term'] + 1e-6)

        # 4. LIQUID_ASSETS_RATIO
        df['liquid_ratio'] = (df['bank_asset_value'] + df['commercial_assets_value']) / \
                             (df['total_assets'] + 1e-6)

        # 5. DEBT_TO_INCOME_DTI
        df['dti'] = df['loan_amount'] / (df['income_annum'] * df['loan_term'] + 1e-6)

        # Normalize new features
        new_features = ['total_assets', 'debt_recovery_ratio', 'repayment_capacity', 'liquid_ratio', 'dti']
        df[new_features] = self.scaler.fit_transform(df[new_features])

        # Add new features to numerical feature list
        self.numerical_features.extend(new_features)

    def select_features(self):
        """Select the most important features for the model"""
        if self.processed_data is None:
            self.process_data()

        df = self.processed_data

        # Select features
        selected_features = [
            'cibil_score',
            'total_assets',
            'debt_recovery_ratio',
            'repayment_capacity',
            'dti',
            'liquid_ratio',
            'income_annum',
            'loan_amount',
            'loan_term',
            'no_of_dependents'
        ]

        # Filter features that exist in the dataset
        selected_features = [f for f in selected_features if f in df.columns]

        print(f"   Selected features ({len(selected_features)}/{len(df.columns) - 1}): {', '.join(selected_features)}")

        # Create feature matrix X and target variable y
        X = df[selected_features]
        y = df[self.target_column]

        return X, y, selected_features

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and test sets (0.2 = 20%)"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
