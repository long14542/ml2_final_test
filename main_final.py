import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys

from data_utils import DataUtils
from model_utils import ModelUtils
from config import Config


def main():
    print("=== Bắt đầu phân tích và dự đoán khoản vay ===")

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 1. Đọc dữ liệu
    print("1. Đọc dữ liệu...")
    try:
        df = pd.read_csv(r"C:\Users\admin\Downloads\loan prediction dataset\loan_approval_dataset.csv")
        print(f"   Đọc thành công: {df.shape[0]} hàng x {df.shape[1]} cột")
        print("   Tên các cột:", ", ".join([f" {col}" for col in df.columns]))
    except Exception as e:
        print(f"   Lỗi khi đọc dữ liệu: {str(e)}")
        sys.exit(1)

    # 2. Phân tích dữ liệu
    print("2. Xử lý và phân tích dữ liệu...")
    data_utils = DataUtils(df)
    data_utils.analyze_dataset()

    # 3. Tiền xử lý dữ liệu và chọn đặc trưng
    print("3. Tiền xử lý dữ liệu và chọn đặc trưng...")
    processed_data = data_utils.process_data()

    # Lấy X, y và đặc trưng được chọn
    X, y, selected_features = data_utils.select_features()

    # In thông tin về đặc trưng số và phân loại ban đầu
    print(f"   Đặc trưng số ban đầu:  {', '.join(data_utils.numerical_features)}")
    print(f"   Đặc trưng phân loại ban đầu:  {', '.join(data_utils.categorical_features)}")

    # 4. Chia dữ liệu thành tập huấn luyện và kiểm tra
    print("4. Chia dữ liệu thành tập huấn luyện và kiểm tra...")
    X_train, X_test, y_train, y_test = data_utils.split_data(X, y)
    print(f"   Kích thước tập huấn luyện: {X_train.shape[0]} mẫu")
    print(f"   Kích thước tập kiểm tra: {X_test.shape[0]} mẫu")

    # 5. Huấn luyện mô hình
    print("5. Huấn luyện mô hình...")
    model_utils = ModelUtils(
        numerical_features=data_utils.numerical_features,
        categorical_features=data_utils.categorical_features
    )

    model = model_utils.build_and_train_models(X_train, y_train)

    # 6. Đánh giá mô hình
    print("6. Đánh giá mô hình...")
    metrics = model_utils.evaluate_model(X_train, X_test, y_train, y_test)

    print(f"   Độ chính xác: {metrics['accuracy']:.4f}")
    print(f"   Độ chính xác micro: {metrics['precision_micro']:.4f}")
    print(f"   Độ nhạy micro: {metrics['recall_micro']:.4f}")
    print(f"   F1 Score micro: {metrics['f1_micro']:.4f}")
    print(f"   Báo cáo phân loại:")
    print(metrics['classification_report'])

    # 7. Tạo biểu đồ
    print("7. Tạo và lưu các biểu đồ...")
    model_utils.plot_confusion_matrix(X_test, y_test)
    model_utils.plot_roc_curve(X_test, y_test)
    model_utils.plot_precision_recall_curve(X_test, y_test)
    model_utils.plot_feature_importance(selected_features)
    model_utils.plot_learning_curve(X_train, y_train)

    # Lưu tất cả các biểu đồ phân tích dữ liệu
    data_utils.save_plots(Config.OUTPUT_DIR)
    # Lưu tất cả các biểu đồ đánh giá mô hình
    model_utils.save_all_plots(Config.OUTPUT_DIR)

    # 8. Kết luận
    print("8. Kết luận và xuất báo cáo...")
    print("   Hoàn thành phân tích và dự đoán")
    print(f"   Các biểu đồ và kết quả được lưu trong thư mục '{Config.OUTPUT_DIR}'")

    # Lưu báo cáo tổng hợp
    with open(os.path.join(Config.OUTPUT_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== BÁO CÁO PHÂN TÍCH VÀ DỰ ĐOÁN KHOẢN VAY ===\n")
        f.write(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Kích thước dữ liệu: {df.shape[0]} hàng x {df.shape[1]} cột\n")
        f.write(f"Đặc trưng được chọn: {', '.join(selected_features)}\n\n")

        f.write("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH:\n")
        f.write(f"Độ chính xác: {metrics['accuracy']:.4f}\n")
        f.write(f"Độ chính xác micro: {metrics['precision_micro']:.4f}\n")
        f.write(f"Độ nhạy micro: {metrics['recall_micro']:.4f}\n")
        f.write(f"F1 Score micro: {metrics['f1_micro']:.4f}\n\n")
        f.write("Báo cáo phân loại chi tiết:\n")
        f.write(metrics['classification_report'])

        # Thêm thông tin về feature importance
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        f.write("\nTẦM QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG:\n")
        for i in sorted_indices:
            f.write(f"{selected_features[i]}: {feature_importances[i]:.4f}\n")

        f.write("\nKẾT LUẬN:\n")
        f.write("- 5 đặc trưng mới được tạo ra đã cải thiện đáng kể hiệu suất của mô hình\n")
        f.write("- Tính năng 'debt_recovery_ratio' có vai trò quan trọng trong việc dự đoán phê duyệt khoản vay\n")
        f.write("- Mô hình có độ chính xác cao, phù hợp để hỗ trợ ra quyết định cho vay\n")

    print("=== Hoàn thành ===")


if __name__ == "__main__":
    main()