import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from config import Config


class DataUtils:
    def __init__(self, raw_data):
        self.raw_data = raw_data.copy()
        # Chuẩn hóa tên cột: loại bỏ khoảng trắng và ký tự đặc biệt
        self.raw_data.columns = [self._clean_column_name(col) for col in self.raw_data.columns]
        self.processed_data = None
        self.numerical_features = []
        self.categorical_features = []
        self.target_column = self._find_target_column()
        self.plots = {}
        self.scaler = RobustScaler()  # Thay đổi từ StandardScaler sang RobustScaler

    def _clean_column_name(self, column_name):
        """Làm sạch tên cột, loại bỏ khoảng trắng và chuyển thành chữ thường"""
        return re.sub(r'[^a-zA-Z0-9_]', '', column_name.strip().lower())

    def _find_target_column(self):
        """Tìm cột mục tiêu là 'loan_status' hoặc tương tự"""
        target_candidates = ['loan_status', 'loanstatus', 'status', 'approved']
        for col in self.raw_data.columns:
            if col.lower() in target_candidates:
                return col
        return None

    def analyze_dataset(self):
        """Phân tích dữ liệu và tạo các biểu đồ thống kê"""
        if self.target_column:
            print(f"   Đã phát hiện cột mục tiêu: {self.target_column}")

            # Phân loại các đặc trưng
            for col in self.raw_data.columns:
                if col == self.target_column or col == 'loan_id':  # Bỏ qua cột ID và cột mục tiêu
                    continue

                if self.raw_data[col].dtype == 'object' or self.raw_data[col].nunique() < 10:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)

            # Tạo các biểu đồ phân tích
            self._create_distribution_plots()
            self._create_correlation_matrix()
            self._create_target_analysis_plots()
        else:
            print("   Không tìm thấy cột mục tiêu phù hợp.")

    def _create_distribution_plots(self):
        """Tạo biểu đồ phân bố cho các đặc trưng số"""
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(self.numerical_features[:6]):  # Chỉ hiển thị tối đa 6 đặc trưng
            plt.subplot(2, 3, i + 1)
            sns.histplot(self.raw_data[feature], kde=True)
            plt.title(f'Phân bố của {feature}')
            plt.tight_layout()

        self.plots['distributions'] = plt.gcf()
        plt.close()

    def _create_correlation_matrix(self):
        """Tạo ma trận tương quan giữa các đặc trưng số"""
        corr_matrix = self.raw_data[self.numerical_features].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Ma trận tương quan giữa các đặc trưng số')
        plt.tight_layout()

        self.plots['correlation'] = plt.gcf()
        plt.close()

    def _create_target_analysis_plots(self):
        """Tạo các biểu đồ phân tích mối quan hệ giữa đặc trưng và biến mục tiêu"""
        # Biểu đồ đếm cho biến mục tiêu
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.target_column, data=self.raw_data)
        plt.title(f'Phân bố biến mục tiêu - {self.target_column}')
        plt.tight_layout()
        self.plots['target_distribution'] = plt.gcf()
        plt.close()

        # Biểu đồ boxplot cho top 4 đặc trưng số quan trọng
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(self.numerical_features[:4]):
            plt.subplot(2, 2, i + 1)
            sns.boxplot(x=self.target_column, y=feature, data=self.raw_data)
            plt.title(f'{feature} theo {self.target_column}')
        plt.tight_layout()
        self.plots['feature_target_relations'] = plt.gcf()
        plt.close()

    def save_plots(self, output_dir=None):
        """Lưu các biểu đồ vào thư mục đầu ra"""
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR

        os.makedirs(output_dir, exist_ok=True)

        for name, fig in self.plots.items():
            fig.savefig(os.path.join(output_dir, f'{name}.png'))

    def process_data(self):
        """Xử lý và làm sạch dữ liệu"""
        df = self.raw_data.copy()

        # Loại bỏ các hàng trùng lặp
        df.drop_duplicates(inplace=True)

        # Loại bỏ cột ID nếu có
        if 'loan_id' in df.columns:
            df.drop('loan_id', axis=1, inplace=True)

        # Xử lý giá trị null
        # Đối với đặc trưng số: điền bằng trung vị
        for feature in self.numerical_features:
            if df[feature].isnull().sum() > 0:
                df[feature].fillna(df[feature].median(), inplace=True)

        # Đối với đặc trưng phân loại: điền bằng giá trị phổ biến nhất
        for feature in self.categorical_features:
            if df[feature].isnull().sum() > 0:
                df[feature].fillna(df[feature].mode()[0], inplace=True)

        # Mã hóa các đặc trưng phân loại
        for feature in self.categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])

        # Mã hóa biến mục tiêu nếu cần
        if df[self.target_column].dtype == 'object':
            le = LabelEncoder()
            df[self.target_column] = le.fit_transform(df[self.target_column])

        # Chuẩn hóa các đặc trưng số
        num_features = [f for f in self.numerical_features if f in df.columns]
        if num_features:
            df[num_features] = self.scaler.fit_transform(df[num_features])

        # Tạo các đặc trưng mới
        self._create_new_features(df)

        self.processed_data = df
        return df

    def _create_new_features(self, df):
        """Tạo 5 đặc trưng mới theo yêu cầu"""
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

        # Chuẩn hóa các đặc trưng mới
        new_features = ['total_assets', 'debt_recovery_ratio', 'repayment_capacity', 'liquid_ratio', 'dti']
        df[new_features] = self.scaler.fit_transform(df[new_features])

        # Thêm các đặc trưng mới vào danh sách đặc trưng số
        self.numerical_features.extend(new_features)

    def select_features(self):
        """Chọn các đặc trưng quan trọng nhất cho mô hình"""
        if self.processed_data is None:
            self.process_data()

        df = self.processed_data

        # Chọn các đặc trưng theo yêu cầu
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

        # Lọc các đặc trưng đã chọn chỉ từ những đặc trưng có trong dữ liệu
        selected_features = [f for f in selected_features if f in df.columns]

        print(
            f"   Đặc trưng được chọn ({len(selected_features)}/{len(df.columns) - 1}):  {', '.join(selected_features)}")

        # Tạo ma trận đặc trưng X và biến mục tiêu y
        X = df[selected_features]
        y = df[self.target_column]

        return X, y, selected_features

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Chia dữ liệu thành tập huấn luyện và kiểm tra"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
