import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

np.random.seed(42)
tf.random.set_seed(42)

# --- ĐỊNH NGHĨA CÁC THAM SỐ CHO DỰ ĐOÁN TRUNG HẠN ---
STABLE_THRESHOLD_PERCENTAGE = 0.1 # Ngưỡng cho lớp "Stable" (ví dụ: 1% thay đổi)
PREDICTION_HORIZON = 60 
# ----------------------------------------------------

# --- Data Loading and Preprocessing (Cập nhật Target cho trung hạn) ---
def load_and_preprocess_data(file_path, stable_percentage, horizon, time_steps=10, train_ratio=0.8):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    list_like_cols = ['Volume_Low', 'Volume_Mid', 'Volume_High', 
                      'BB_Width_Low', 'BB_Width_Mid', 'BB_Width_High',
                      'ATR_Low', 'ATR_Mid', 'ATR_High']
    for col in list_like_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: float(str(x).strip('[]')) if isinstance(x, (str, list)) and str(x).strip('[]') != '' and str(x).strip('[]') != '.' else np.nan)

    numerical_cols = [col for col in df.columns if col != 'Date']
    for col in numerical_cols:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '', regex=False).astype(float)
            except AttributeError: pass
            except ValueError:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    actual_news_cols = []
    # Giả sử tên cột sentiment/news trong file CSV của bạn
    if 'Sentiment_Score' in df.columns and 'Sentiment_Score' not in actual_news_cols: actual_news_cols.append('Sentiment_Score')
    elif 'mean_sentiment' in df.columns and 'mean_sentiment' not in actual_news_cols: actual_news_cols.append('mean_sentiment')
    if 'news_count' in df.columns and 'news_count' not in actual_news_cols: actual_news_cols.append('news_count')
    if 'news_count_daily' in df.columns and not actual_news_cols and 'news_count' not in actual_news_cols :
        actual_news_cols.append('news_count_daily')

    # Sử dụng tên cột từ file bạn cung cấp cho việc lag (VIXCLS, FEDFUNDS, etc.)
    # daily_policy_index -> USEPUINDXD
    # News_Based_Policy_Uncert_Index -> NEWSEPU
    cols_to_lag_explicitly_base = ['Gold_Price', 'VIXCLS', 'FEDFUNDS', 
                                   'daily_policy_index', 'News_Based_Policy_Uncert_Index'] 
    cols_to_lag_explicitly = [col for col in cols_to_lag_explicitly_base if col in df.columns] + actual_news_cols
    cols_to_lag_explicitly = list(dict.fromkeys([col for col in cols_to_lag_explicitly if col in df.columns]))
    
    print(f"Các cột được xác định để lấy giá trị trễ (t-1): {cols_to_lag_explicitly}")

    for col in cols_to_lag_explicitly:
        df[f'{col}_lag1'] = df[col].shift(1)

    vix_col_for_change = None
    if 'VIXCLS' in df.columns: vix_col_for_change = 'VIXCLS'
    elif 'VIX_Close' in df.columns: vix_col_for_change = 'VIX_Close'
    if vix_col_for_change: df[f'{vix_col_for_change}_Change'] = df[vix_col_for_change].diff() # Đổi tên VIX_Change
    else: 
        df['VIX_Change_Fallback'] = np.nan # Tạo cột fallback nếu không có VIX
        print("Cảnh báo: Không tìm thấy cột VIX (VIXCLS hoặc VIX_Close) để tạo VIX_Change.")


    if 'Gold_Price' in df.columns: df['Gold_Price_Change'] = df['Gold_Price'].diff()
    else: df['Gold_Price_Change'] = np.nan 
    
    # --- CẬP NHẬT ĐỊNH NGHĨA TARGET CHO TRUNG HẠN ---
    df['Future_Close'] = df['Close'].shift(-horizon) # Lấy giá đóng cửa của 'horizon' ngày sau
    df['Price_Change_Pct_Future'] = (df['Future_Close'] - df['Close']) / (df['Close'] + 1e-9) 

    def define_target_multiclass(change_pct):
        if pd.isna(change_pct): return np.nan 
        if change_pct > stable_percentage: return 2 # Increase
        elif change_pct < -stable_percentage: return 0 # Decrease
        else: return 1 # Stable
    df['Target'] = df['Price_Change_Pct_Future'].apply(define_target_multiclass)
    df.drop(columns=['Future_Close', 'Price_Change_Pct_Future'], inplace=True)
    # ---------------------------------------------

    df['Lagged_Target_1'] = df['Target'].shift(1) 
    
    df.dropna(subset=['Target'], inplace=True) # Loại bỏ các hàng không có Target (cuối DataFrame)
    df['Target'] = df['Target'].astype(int)

    df_for_feature_selection = df.copy()
    
    # Sử dụng excluded_cols_from_features từ code bạn cung cấp
    excluded_cols_from_features =  ['Date', 'Target', 'RSI', 'Volume','BB', 'ATR', 'Close'] 
    
    original_cols_replaced_by_lag = set(cols_to_lag_explicitly)
    feature_cols = []
    for col in df_for_feature_selection.columns: 
        if col in excluded_cols_from_features: continue
        lagged_version_name = f"{col}_lag1"
        if col in original_cols_replaced_by_lag:
            if lagged_version_name in df_for_feature_selection.columns:
                if lagged_version_name not in feature_cols: feature_cols.append(lagged_version_name)
        elif col.endswith('_lag1') and col.replace('_lag1','') in original_cols_replaced_by_lag:
            if col not in feature_cols: feature_cols.append(col)
        else: 
            if col not in feature_cols: feature_cols.append(col)
    feature_cols = list(dict.fromkeys(feature_cols)) 
    
    print(f"Các features sẽ được sử dụng: {len(feature_cols)} features")
    if not feature_cols: raise ValueError("Không còn feature nào sau khi xử lý.")

    df_subset_for_processing = df[feature_cols + ['Target']].copy()
    df_subset_for_processing[feature_cols] = df_subset_for_processing[feature_cols].ffill().bfill()
    for col in feature_cols:
        if df_subset_for_processing[col].isnull().any(): 
            df_subset_for_processing[col] = df_subset_for_processing[col].fillna(0)
    for col in feature_cols:
        df_subset_for_processing[col] = pd.to_numeric(df_subset_for_processing[col], errors='coerce').fillna(0)
        if not np.issubdtype(df_subset_for_processing[col].dtype, np.number):
             raise ValueError(f"Cột {col} chứa dữ liệu không phải số: {df_subset_for_processing[col].head()}")
    
    features_data = df_subset_for_processing[feature_cols].values
    target_data = df_subset_for_processing['Target'].values

    train_size_index = int(len(features_data) * train_ratio)
    train_features_unscaled = features_data[:train_size_index]
    test_features_unscaled = features_data[train_size_index:]
    y_target_for_train_sequences = target_data[:train_size_index]
    y_target_for_test_sequences = target_data[train_size_index:]

    scaler = MinMaxScaler()
    scaled_train_features = scaler.fit_transform(train_features_unscaled)
    if np.isnan(scaled_train_features).any(): scaled_train_features = np.nan_to_num(scaled_train_features)
    scaled_test_features = scaler.transform(test_features_unscaled)
    if np.isnan(scaled_test_features).any(): scaled_test_features = np.nan_to_num(scaled_test_features)

    def create_sequences(features_scaled, original_target_array, time_steps_local):
        X_seq, y_seq = [], []
        num_features_in_input_arr = features_scaled.shape[1] if features_scaled.ndim > 1 and features_scaled.shape[1] > 0 else len(feature_cols)
        if len(features_scaled) <= time_steps_local:
            return np.array([]).reshape(0, time_steps_local, num_features_in_input_arr), np.array([])
        for i in range(len(features_scaled) - time_steps_local):
            X_seq.append(features_scaled[i : i + time_steps_local])
            y_seq.append(original_target_array[i + time_steps_local])
        if not X_seq:
             return np.array([]).reshape(0, time_steps_local, num_features_in_input_arr), np.array([])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = create_sequences(scaled_train_features, y_target_for_train_sequences, time_steps)
    X_test, y_test = create_sequences(scaled_test_features, y_target_for_test_sequences, time_steps)
    
    num_actual_features = len(feature_cols)
    if X_train.ndim == 1 and X_train.shape[0] == 0: X_train = X_train.reshape(0, time_steps, num_actual_features) 
    if X_test.ndim == 1 and X_test.shape[0] == 0: X_test = X_test.reshape(0, time_steps, num_actual_features)
    if X_train.shape[0] == 0 and (X_train.ndim < 3 or X_train.shape[2] != num_actual_features) : 
        X_train = np.empty((0, time_steps, num_actual_features))
    if X_test.shape[0] == 0 and (X_test.ndim < 3 or X_test.shape[2] != num_actual_features) : 
        X_test = np.empty((0, time_steps, num_actual_features))

    if X_train.shape[0] == 0 or X_test.shape[0] == 0 :
        raise ValueError(f"Không đủ dữ liệu để tạo sequences với time_steps={time_steps}. ")
    if X_train.shape[2] != num_actual_features:
        raise ValueError(f"Số features X_train ({X_train.shape[2]}) không khớp feature_cols ({num_actual_features}).")
                         
    return X_train, X_test, y_train, y_test, scaler, feature_cols, df_for_feature_selection


# --- Định nghĩa không gian tìm kiếm (CÓ THỂ ĐIỀU CHỈNH time_steps CHO TRUNG HẠN) ---
space_optimized_single_lstm = [
    Integer(10, 40, name='time_steps'), # Tăng phạm vi time_steps, ví dụ: 10-40 ngày
    Integer(30, 150, name='lstm_units1'),     
    Real(0.1, 0.5, name='dropout_rate1'),     
    Integer(15, 120, name='dense_units'),      
    Categorical(['relu', 'tanh', 'elu'], name='dense_activation'), 
    Real(1e-5, 1e-2, "log-uniform", name='learning_rate'), 
    Categorical([32, 64, 128], name='batch_size'),
    Categorical([True, False], name='use_batch_norm_lstm'), 
    Categorical([True, False], name='use_batch_norm_dense'),
    Real(0.0, 0.005, name='l1_reg_lstm'), 
    Real(0.0, 0.005, name='l2_reg_lstm'),
    Real(0.0, 0.005, name='l1_reg_dense'),
    Real(0.0, 0.005, name='l2_reg_dense')
]

FILE_PATH = 'data/merged_with_policy.csv' 
BASE_RESULTS_DIR = "experiment_results_medium_term_v1" # Thư mục mới

fitness_call_count = 0
current_run_results_dir = "" 

@use_named_args(space_optimized_single_lstm) 
def fitness(time_steps, lstm_units1, dropout_rate1, 
            dense_units, dense_activation, learning_rate, batch_size,
            use_batch_norm_lstm, use_batch_norm_dense,
            l1_reg_lstm, l2_reg_lstm, l1_reg_dense, l2_reg_dense):
    global fitness_call_count
    fitness_call_count += 1
    print(f"\nLần gọi hàm fitness thứ: {fitness_call_count}")
    current_params = {param.name: value for param, value in zip(space_optimized_single_lstm, list(locals().values())[:len(space_optimized_single_lstm)])}
    for key, val in current_params.items():
        if isinstance(val, float): current_params[key] = round(val, 6)
    print(f"Đang thử nghiệm với các tham số (medium-term LSTM v1): {current_params}")

    try:
        current_time_steps = int(time_steps)
        X_train, X_test, y_train, y_test, _, current_feature_cols, _ = load_and_preprocess_data(
            FILE_PATH, 
            stable_percentage=STABLE_THRESHOLD_PERCENTAGE, 
            horizon=PREDICTION_HORIZON, # Thêm horizon
            time_steps=current_time_steps
        )
        if X_train.shape[0] == 0 or X_test.shape[0] == 0: raise ValueError("X_train/X_test rỗng.")
        if X_train.shape[1] != current_time_steps: raise ValueError(f"Shape X_train time_steps không khớp.")
        if X_train.shape[2] == 0 or X_train.shape[2] != len(current_feature_cols): 
            raise ValueError(f"Số features X_train ({X_train.shape[2]}) không khớp ({len(current_feature_cols)}).")
    except ValueError as e:
        print(f"Lỗi khi xử lý dữ liệu: {e}. Trả về giá trị loss lớn.")
        return 10.0 
    except Exception as e_gen:
        print(f"Lỗi không xác định khi xử lý dữ liệu: {e_gen}. Trả về giá trị loss lớn.")
        return 10.0

    actual_lstm_units1 = int(lstm_units1)
    actual_dense_units = int(dense_units)
    current_batch_size = int(batch_size)

    model = Sequential(name=f"MediumTermLSTM_FitnessCall_{fitness_call_count}")
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2]), name="input_layer"))
    model.add(LSTM(actual_lstm_units1, return_sequences=False, name="lstm_1",
                   kernel_regularizer=l1_l2(l1=l1_reg_lstm, l2=l2_reg_lstm)))
    if use_batch_norm_lstm: model.add(BatchNormalization(name="bn_lstm"))
    model.add(Dropout(dropout_rate1, name="dropout_1"))
    model.add(Dense(actual_dense_units, activation=dense_activation, name="dense_1",
                    kernel_regularizer=l1_l2(l1=l1_reg_dense, l2=l2_reg_dense)))
    if use_batch_norm_dense: model.add(BatchNormalization(name="bn_dense"))
    model.add(Dense(3, activation='softmax', name="output_layer"))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=0) 
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max', verbose=0)

    if y_train.shape[0] == 0 or y_test.shape[0] == 0:
        print("Lỗi: y_train hoặc y_test rỗng. Không thể huấn luyện.")
        return 10.0
        
    class_weights_for_fit = None
    unique_classes_train_fit, counts_classes_train_fit = np.unique(y_train, return_counts=True)
    if len(unique_classes_train_fit) >= 2 : 
        weights_fit = compute_class_weight('balanced', classes=unique_classes_train_fit, y=y_train)
        class_weights_for_fit = dict(zip(unique_classes_train_fit, weights_fit))
    
    model.fit( 
        X_train, y_train, epochs=70, batch_size=current_batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr], verbose=0,
        class_weight=class_weights_for_fit 
    )
    
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Kết quả: Val Accuracy = {accuracy:.4f}")
    tf.keras.backend.clear_session() 
    return -accuracy

# --- Các hàm tiện ích để lưu kết quả (Cập nhật tên file cho phù hợp) ---
def save_bayesian_opt_summary(results_dir, skopt_result, param_space_used):
    summary_path = os.path.join(results_dir, "bayesian_optimization_summary.txt")
    best_params_dict = {param.name: value for param, value in zip(param_space_used, skopt_result.x)}
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Bayesian Optimization Summary (Medium-Term LSTM V1, Horizon={PREDICTION_HORIZON}) ---\n")
        f.write(f"Best Objective Function Value (-Validation Accuracy): {skopt_result.fun:.4f}\n")
        f.write("Best Hyperparameters Found:\n")
        # ... (phần còn lại giống như trước)
        for param_name, value in best_params_dict.items():
            if isinstance(value, float): value_str = f"{value:.6f}"
            else: value_str = str(value)
            f.write(f"  {param_name}: {value_str}\n")
        f.write("\n--- Optimization Process (All Trials) ---\n")
        header = "Trial | -Val_Accuracy | " + " | ".join([dim.name for dim in param_space_used]) + "\n"
        f.write(header)
        for i, (params_trial_values, func_val) in enumerate(zip(skopt_result.x_iters, skopt_result.func_vals)):
            params_str_list = []
            for val_idx, val_trial in enumerate(params_trial_values):
                param_name_trial = param_space_used[val_idx].name
                if isinstance(val_trial, float): params_str_list.append(f"{param_name_trial}={val_trial:.5f}")
                else: params_str_list.append(f"{param_name_trial}={str(val_trial)}")
            params_str = ", ".join(params_str_list)
            f.write(f"{i+1:03d}   | {func_val:14.4f} | {params_str}\n")

    print(f"Tóm tắt Bayesian Optimization đã được lưu vào: {summary_path}")
    try:
        plot_convergence(skopt_result)
        plt.title(f"Convergence Plot (Medium-Term LSTM, H={PREDICTION_HORIZON})")
        plt.savefig(os.path.join(results_dir, "bayesian_opt_convergence_medium_term.png"))
        plt.close()
        dimension_names_for_plot = [dim.name for dim in param_space_used if isinstance(dim, (Integer, Real, Categorical))]
        plot_dims = [name for name in ['time_steps', 'lstm_units1', 'learning_rate', 'dense_activation','use_batch_norm_lstm'] if name in dimension_names_for_plot]
        if not plot_dims and dimension_names_for_plot : plot_dims = dimension_names_for_plot[:min(5, len(dimension_names_for_plot))]
        if plot_dims:
            plot_objective(skopt_result, dimensions=plot_dims) 
            plt.suptitle(f"Objective Function Landscape (Medium-Term LSTM, H={PREDICTION_HORIZON} - Partial)")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(results_dir, "bayesian_opt_objective_landscape_medium_term.png"))
            plt.close()
        print(f"Biểu đồ Bayesian Optimization đã được lưu trong thư mục: {results_dir}")
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ Bayesian Optimization: {e}")


def save_final_model_evaluation(results_dir, metrics, filename_suffix=""):
    metrics_path = os.path.join(results_dir, f"final_model_evaluation_metrics{filename_suffix}.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Final Model Evaluation Metrics (Medium-Term LSTM, H={PREDICTION_HORIZON}{filename_suffix}) ---\n")
        # ... (phần còn lại giống như trước)
        for metric_name, value in metrics.items():
            if isinstance(value, float): f.write(f"{metric_name}: {value:.4f}\n")
            else: f.write(f"{metric_name}: {value}\n")
    print(f"Các chỉ số đánh giá mô hình cuối cùng đã được lưu vào: {metrics_path}")


def plot_training_history_custom(history, results_dir, call_info=""): # Giữ nguyên
    # ... (Nội dung hàm giữ nguyên)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history: plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss {call_info}')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history: plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy {call_info}')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plot_filename = f'training_history_{call_info.replace(" ", "_").replace(":", "")}.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()
    print(f"Biểu đồ lịch sử huấn luyện đã được lưu vào: {os.path.join(results_dir, plot_filename)}")

def plot_confusion_matrix_custom(y_true, y_pred, results_dir, class_names, call_info=""): # Giữ nguyên
    # ... (Nội dung hàm giữ nguyên)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 8)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix {call_info}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plot_filename = f'confusion_matrix_{call_info.replace(" ", "_").replace(":", "")}.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()
    print(f"Biểu đồ confusion matrix đã được lưu vào: {os.path.join(results_dir, plot_filename)}")


# --- Main Function ---
def main_with_bayesian_optimization():
    global current_run_results_dir
    if not os.path.exists(BASE_RESULTS_DIR): os.makedirs(BASE_RESULTS_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_results_dir = os.path.join(BASE_RESULTS_DIR, f"run_medium_term_h{PREDICTION_HORIZON}_{timestamp}") 
    os.makedirs(current_run_results_dir)
    print(f"Kết quả sẽ được lưu vào: {current_run_results_dir}")

    print(f"Bắt đầu Bayesian Optimization (Medium-Term LSTM V1, Horizon={PREDICTION_HORIZON})...")
    skopt_result = gp_minimize(
        func=fitness, dimensions=space_optimized_single_lstm, 
        n_calls=25, # Số lần gọi, có thể điều chỉnh
        n_initial_points=7, 
        random_state=42,
        verbose=True 
    )

    print(f"\n--- Kết quả Bayesian Optimization (Medium-Term LSTM V1, Horizon={PREDICTION_HORIZON}) ---")
    save_bayesian_opt_summary(current_run_results_dir, skopt_result, space_optimized_single_lstm)
    best_params_values = skopt_result.x
    best_params_dict = {param.name: value for param, value in zip(space_optimized_single_lstm, best_params_values)}
    print(f"Siêu tham số tốt nhất: {best_params_dict}")
    print(f"Giá trị mục tiêu tốt nhất (-Val Accuracy): {skopt_result.fun:.4f}")

    print(f"\n--- Huấn luyện mô hình cuối cùng (Medium-Term LSTM V1, Horizon={PREDICTION_HORIZON}) ---")
    best_time_steps = int(best_params_dict['time_steps'])
    
    class_weights_final_model = None
    try:
        X_train, X_test, y_train, y_test, final_scaler, feature_cols, df_full_processed = load_and_preprocess_data(
            FILE_PATH, 
            stable_percentage=STABLE_THRESHOLD_PERCENTAGE, 
            horizon=PREDICTION_HORIZON, # Thêm horizon
            time_steps=best_time_steps
        )
        if X_train.shape[0] == 0 or X_test.shape[0] == 0: raise ValueError("X_train/X_test rỗng.")
        if X_train.shape[2] == 0: raise ValueError(f"Số features X_train bằng 0.")

        if y_train.shape[0] > 0:
            unique_classes, counts_classes = np.unique(y_train, return_counts=True)
            print(f"Phân phối lớp trong y_train (để tính class_weight): {dict(zip(unique_classes, counts_classes))}")
            if len(unique_classes) >= 2: 
                weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
                class_weights_final_model = dict(zip(unique_classes, weights))
                # Bạn có thể điều chỉnh thủ công class_weights_final_model ở đây nếu cần
                # Ví dụ: if 0 in class_weights_final_model: class_weights_final_model[0] *= 1.5 
                print(f"Class weights ('balanced') được tính cho mô hình cuối cùng: {class_weights_final_model}")
            else: print("Không đủ số lớp trong y_train để tính class_weight cân bằng.")
        else: print("y_train rỗng, không thể tính class_weight.")
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu hoặc tính class_weight: {e}")
        return

    actual_final_lstm_units1 = int(best_params_dict['lstm_units1'])
    actual_final_dense_units = int(best_params_dict['dense_units'])
    final_batch_size = int(best_params_dict['batch_size'])
    
    final_model = Sequential(name=f"MediumTerm_LSTM_Final_H{PREDICTION_HORIZON}")
    final_model.add(Input(shape=(X_train.shape[1], X_train.shape[2]), name="input_layer_final"))
    final_model.add(LSTM(actual_final_lstm_units1, return_sequences=False, name="lstm_1_final",
                         kernel_regularizer=l1_l2(l1=best_params_dict['l1_reg_lstm'], l2=best_params_dict['l2_reg_lstm'])))
    if best_params_dict['use_batch_norm_lstm']:
        final_model.add(BatchNormalization(name="bn_lstm_final"))
    final_model.add(Dropout(best_params_dict['dropout_rate1'], name="dropout_1_final"))
    final_model.add(Dense(actual_final_dense_units, activation=best_params_dict['dense_activation'], name="dense_1_final",
                          kernel_regularizer=l1_l2(l1=best_params_dict['l1_reg_dense'], l2=best_params_dict['l2_reg_dense'])))
    if best_params_dict['use_batch_norm_dense']:
        final_model.add(BatchNormalization(name="bn_dense_final"))
    final_model.add(Dense(3, activation='softmax', name="output_layer_final"))
    
    final_optimizer = Adam(learning_rate=best_params_dict['learning_rate'])
    final_model.compile(optimizer=final_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    summary_list = []; final_model.summary(print_fn=lambda x: summary_list.append(x))
    model_summary_str = "\n".join(summary_list); print(model_summary_str)
    summary_file_path = os.path.join(current_run_results_dir, f"final_model_summary_medium_term_h{PREDICTION_HORIZON}.txt")
    with open(summary_file_path, 'w', encoding='utf-8') as f: f.write(model_summary_str)
    print(f"Model summary đã lưu vào: {summary_file_path}")

    early_stopping_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    reduce_lr_final = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1)
    
    print(f"Huấn luyện mô hình cuối cùng với class_weight: {class_weights_final_model}")
    history = final_model.fit( 
        X_train, y_train, epochs=120, batch_size=final_batch_size,
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping_final, reduce_lr_final], 
        verbose=1,
        class_weight=class_weights_final_model
    )
    
    y_pred_prob = final_model.predict(X_test)
    y_pred_final = np.argmax(y_pred_prob, axis=1)
    
    print("Kiểm tra y_pred_final sau khi argmax:")
    unique_preds_check, counts_preds_check = np.unique(y_pred_final, return_counts=True)
    print(f"Các giá trị duy nhất trong y_pred_final (kiểm tra): {unique_preds_check}")
    print(f"Số lượng tương ứng (kiểm tra): {counts_preds_check}")

    avg_method = 'macro' 
    metrics_dict = { 
        "Accuracy": accuracy_score(y_test, y_pred_final),
        "Precision_macro": precision_score(y_test, y_pred_final, average=avg_method, zero_division=0),
        "Recall_macro": recall_score(y_test, y_pred_final, average=avg_method, zero_division=0),
        "F1-Score_macro": f1_score(y_test, y_pred_final, average=avg_method, zero_division=0)
    }
    precision_per_class = precision_score(y_test, y_pred_final, average=None, zero_division=0, labels=[0,1,2])
    recall_per_class = recall_score(y_test, y_pred_final, average=None, zero_division=0, labels=[0,1,2])
    f1_per_class = f1_score(y_test, y_pred_final, average=None, zero_division=0, labels=[0,1,2])
    class_labels_map = {0: "Decrease", 1: "Stable", 2: "Increase"}
    for i in range(len(class_labels_map)): 
        if i < len(precision_per_class): metrics_dict[f"Precision_{class_labels_map.get(i)}"] = precision_per_class[i]
        if i < len(recall_per_class): metrics_dict[f"Recall_{class_labels_map.get(i)}"] = recall_per_class[i]
        if i < len(f1_per_class): metrics_dict[f"F1-Score_{class_labels_map.get(i)}"] = f1_per_class[i]

    print(f"\nĐánh giá Mô hình Cuối Cùng (Medium-Term, H={PREDICTION_HORIZON}, average='{avg_method}'):")
    for name, value in metrics_dict.items(): 
        if isinstance(value, float): print(f"{name}: {value:.4f}")
        else: print(f"{name}: {value}")
    save_final_model_evaluation(current_run_results_dir, metrics_dict, filename_suffix=f"_medium_term_h{PREDICTION_HORIZON}")
    
    plot_training_history_custom(history, current_run_results_dir, call_info=f'final_model_medium_term_h{PREDICTION_HORIZON}')
    class_names = ['Decrease', 'Stable', 'Increase']
    plot_confusion_matrix_custom(y_test, y_pred_final, current_run_results_dir, class_names=class_names, call_info=f'final_model_medium_term_h{PREDICTION_HORIZON}')
    
    model_save_path = os.path.join(current_run_results_dir, f'lstm_medium_term_h{PREDICTION_HORIZON}_model.keras')
    final_model.save(model_save_path)
    print(f"\nMô hình cuối cùng đã lưu vào: {model_save_path}")
    
    # Dự đoán cho 'horizon' ngày tới (dựa trên dữ liệu cuối cùng có sẵn)
    final_features_data = df_full_processed[feature_cols].values
    if len(final_features_data) < best_time_steps:
        print(f"Không đủ dữ liệu cuối để dự đoán với time_steps={best_time_steps}")
    else:
        last_data_points_unscaled = final_features_data[-best_time_steps:]
        if last_data_points_unscaled.shape[1] != final_scaler.n_features_in_:
            print(f"Lỗi: Số features dữ liệu cuối ({last_data_points_unscaled.shape[1]}) không khớp scaler ({final_scaler.n_features_in_}).")
        else:
            last_sequence_scaled = final_scaler.transform(last_data_points_unscaled)
            last_sequence_reshaped = last_sequence_scaled.reshape(1, best_time_steps, len(feature_cols))
            next_trend_probs_all_classes = final_model.predict(last_sequence_reshaped)[0]
            next_trend_class_idx = np.argmax(next_trend_probs_all_classes)
            next_trend = class_names[next_trend_class_idx]
            
            last_actual_date = df_full_processed['Date'].iloc[-1] # Ngày cuối cùng trong dữ liệu đã xử lý feature
            prediction_summary_path = os.path.join(current_run_results_dir, f"next_horizon_prediction_h{PREDICTION_HORIZON}.txt")
            prediction_text = (f"Dự đoán xu hướng cho {PREDICTION_HORIZON} ngày giao dịch tới (kể từ sau {last_actual_date.strftime('%Y-%m-%d')}): {next_trend}\n"
                               f"Xác suất các lớp (Decrease, Stable, Increase): {np.round(next_trend_probs_all_classes, 4)}")
            print(f"\n{prediction_text}")
            with open(prediction_summary_path, 'w', encoding='utf-8') as f: f.write(prediction_text)
            print(f"Dự đoán cho trung hạn tiếp theo đã được lưu vào: {prediction_summary_path}")

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Đã tạo thư mục 'data'. Vui lòng đặt file merged_with_policy.csv vào đó.")
    if os.path.exists(FILE_PATH):
        main_with_bayesian_optimization()
    else:
        print(f"Lỗi: Không tìm thấy file dataset tại '{FILE_PATH}'.")