import os

class Config:
    # Create relative paths for the project
    # Root directory containing the project
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path to dataset
    DATA_FILE_PATH = os.path.join(ROOT_DIR, "loan_approval_dataset.csv")

    # Folder with output results
    OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

    # Model configuration
    MODEL_CONFIG = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 20,
            "max_features": "sqrt",
            "random_state": 42,
            "class_weight": "balanced"
        },
        # Additional model configurations can be added here
    }

    # Configuration for splitting data
    TRAIN_TEST_SPLIT = {
        "test_size": 0.2,
        "random_state": 42
    }

    # Configuration for plotting graphs
    VISUALIZATION = {
        "figure_size_large": (12, 8),
        "figure_size_medium": (10, 6),
        "figure_size_small": (8, 6),
        "dpi": 100,
        "cmap": "coolwarm"
    }

    # Other parameters
    RANDOM_STATE = 42
    NUM_CV_FOLDS = 5

    # Add 5 new features
    NEW_FEATURES_CONFIG = {
        "total_assets": {
            "description": "Total value of the borrower's available assets."
        },
        "debt_recovery_ratio": {
            "description": "Ratio of recoverable collateral assets compared to the loan."
        },
        "repayment_capacity": {
            "description": "Repayment capacity index combining income and assets."
        },
        "liquid_ratio": {
            "description": "Ratio of highly liquid assets to total assets."
        },
        "dti": {
            "description": "Debt-to-total accumulated income ratio."
        }
    }
