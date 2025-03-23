import os


class Config:
    # Tạo các đường dẫn tương đối cho dự án
    # Thư mục gốc chứa dự án
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path to dataset
    DATA_FILE_PATH = os.path.join(ROOT_DIR, "loan_approval_dataset.csv")

    # folder with output results
    OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

    # Config model
    MODEL_CONFIG = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 20,
            "max_features": "sqrt",
            "random_state": 42,
            "class_weight": "balanced"
        },
        # Có thể thêm cấu hình cho các mô hình khác ở đây
    }

    # config for spliting data
    TRAIN_TEST_SPLIT = {
        "test_size": 0.2,
        "random_state": 42
    }

    # config for plot graph
    VISUALIZATION = {
        "figure_size_large": (12, 8),
        "figure_size_medium": (10, 6),
        "figure_size_small": (8, 6),
        "dpi": 100,
        "cmap": "coolwarm"
    }

    # Các thông số khác
    RANDOM_STATE = 42
    NUM_CV_FOLDS = 5

    # Add 5 new features
    NEW_FEATURES_CONFIG = {
        "total_assets": {
            "description": "Tổng giá trị tài sản hiện có của người vay."
        },
        "debt_recovery_ratio": {
            "description": "Tỷ lệ tài sản thế chấp có thể thu hồi so với khoản vay."
        },
        "repayment_capacity": {
            "description": "Chỉ số khả năng trả nợ kết hợp thu nhập và tài sản."
        },
        "liquid_ratio": {
            "description": "Tỷ lệ tài sản thanh khoản cao trong tổng tài sản."
        },
        "dti": {
            "description": "Tỷ lệ nợ trên tổng thu nhập tích lũy."
        }
    }
