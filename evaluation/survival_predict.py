import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from tqdm import tqdm
import json
import glob
from pathlib import Path

# ========================
# 1. 설정 및 경로
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 경로 설정
TSV_PATH = "LUAD_clinical.tsv"
FS_FEATURE_DIR = "features/FS"
FFPE_FEATURE_DIR = "features/FFPE"
OUTPUT_DIR = "survival_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

K_FOR_KNN = 8

# ========================
# k-NN 그래프 직접 구현 (torch-cluster 대체)
# ========================

def manual_knn_graph(x, k):
    """
    수동 k-NN 그래프 구성 (torch-cluster 없이)
    
    Args:
        x: (num_nodes, feature_dim) 노드 특징
        k: k-nearest neighbors
    
    Returns:
        edge_index: (2, num_edges) 엣지 인덱스
    """
    num_nodes = x.size(0)
    
    # 모든 노드 쌍 간의 거리 계산 (L2 distance)
    # (num_nodes, num_nodes)
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # Broadcasting
    dist = torch.sum(diff ** 2, dim=2)  # Squared L2 distance
    
    # 각 노드에 대해 k개의 nearest neighbors 찾기
    # topk는 가장 작은 거리를 찾음
    _, indices = torch.topk(dist, k=min(k+1, num_nodes), largest=False, dim=1)
    
    # 자기 자신 제외 (거리 0인 것)
    indices = indices[:, 1:k+1]  # (num_nodes, k)
    
    # Edge index 구성
    src_nodes = torch.arange(num_nodes, device=x.device).unsqueeze(1).repeat(1, k)
    edge_index = torch.stack([src_nodes.flatten(), indices.flatten()], dim=0)
    
    return edge_index

# ========================
# C-Index 직접 구현
# ========================

def concordance_index(event_times, predicted_scores, event_observed):
    """Concordance Index (C-Index) 계산"""
    n = len(event_times)
    
    if n == 0:
        return 0.5
    
    concordant = 0
    discordant = 0
    tied_risk = 0
    
    for i in range(n):
        if not event_observed[i]:
            continue
            
        for j in range(n):
            if event_times[i] < event_times[j]:
                if predicted_scores[i] > predicted_scores[j]:
                    concordant += 1
                elif predicted_scores[i] < predicted_scores[j]:
                    discordant += 1
                else:
                    tied_risk += 1
            elif event_times[i] == event_times[j] and event_observed[j]:
                if predicted_scores[i] == predicted_scores[j]:
                    tied_risk += 1
    
    total = concordant + discordant + tied_risk
    
    if total == 0:
        return 0.5
    
    c_index = (concordant + 0.5 * tied_risk) / total
    
    return c_index

def time_dependent_auc(train_event_times, train_event_observed, 
                       test_event_times, test_event_observed, 
                       test_risk_scores, time_point):
    """Time-dependent AUC 계산"""
    case_indices = (test_event_times <= time_point) & test_event_observed
    control_indices = test_event_times > time_point
    
    if np.sum(case_indices) == 0 or np.sum(control_indices) == 0:
        return None
    
    case_scores = test_risk_scores[case_indices]
    control_scores = test_risk_scores[control_indices]
    
    concordant = 0
    total = 0
    
    for case_score in case_scores:
        for control_score in control_scores:
            if case_score > control_score:
                concordant += 1
            elif case_score == control_score:
                concordant += 0.5
            total += 1
    
    if total == 0:
        return None
    
    return concordant / total

# ========================
# 2. 데이터 전처리
# ========================

def prepare_survival_data(tsv_path):
    """TSV에서 생존 데이터 준비"""
    df = pd.read_csv(tsv_path, sep='\t')
    
    survival_df = df.groupby('case_submitter_id').agg({
        'days_to_death': 'first',
        'days_to_last_follow_up': 'first',
        'vital_status': 'first',
        'age_at_index': 'first'
    }).reset_index()
    
    survival_df['time'] = survival_df.apply(
        lambda row: row['days_to_death'] if pd.notna(row['days_to_death']) and row['days_to_death'] != '--'
        else row['days_to_last_follow_up'] if pd.notna(row['days_to_last_follow_up']) and row['days_to_last_follow_up'] != '--'
        else np.nan, axis=1
    )
    
    survival_df['event'] = survival_df['vital_status'].apply(
        lambda x: 1 if x == 'Dead' else 0
    )
    
    survival_df = survival_df[survival_df['time'].notna()].copy()
    survival_df['time'] = pd.to_numeric(survival_df['time'], errors='coerce')
    survival_df = survival_df[survival_df['time'] > 0].copy()
    
    print(f"Total patients with valid survival data: {len(survival_df)}")
    print(f"Events (deaths): {survival_df['event'].sum()}")
    print(f"Censored (alive): {(1 - survival_df['event']).sum()}")
    
    return survival_df

def find_feature_folders(feature_root_dir, survival_df):
    """Feature 디렉토리에서 환자별 .npy 파일들 찾기"""
    matched_data = []
    
    for patient_id in survival_df['case_submitter_id'].unique():
        patient_dir = os.path.join(feature_root_dir, patient_id)
        
        if not os.path.exists(patient_dir):
            continue
        
        image_folders = [d for d in os.listdir(patient_dir) 
                        if os.path.isdir(os.path.join(patient_dir, d))]
        
        for img_folder in image_folders:
            npy_files = glob.glob(os.path.join(patient_dir, img_folder, "*.npy"))
            npy_files = [f for f in npy_files if "__MACOSX" not in f and "._" not in os.path.basename(f)]
            
            if len(npy_files) > 0:
                patient_surv = survival_df[survival_df['case_submitter_id'] == patient_id].iloc[0]
                matched_data.append({
                    'case_id': patient_id,
                    'feature_folder': os.path.join(patient_dir, img_folder),
                    'num_patches': len(npy_files),
                    'time': patient_surv['time'],
                    'event': patient_surv['event']
                })
    
    return pd.DataFrame(matched_data)

# ========================
# 3. PyG Dataset 클래스 (수정됨 - manual_knn_graph 사용)
# ========================

class WsiSurvivalGraphDataset(Dataset):
    def __init__(self, data_df, device_for_knn):
        super(WsiSurvivalGraphDataset, self).__init__()
        self.data = data_df.reset_index(drop=True)
        self.device_for_knn = device_for_knn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        case_id = row['case_id']
        feature_folder = row['feature_folder']
        time = torch.tensor([row['time']], dtype=torch.float)
        event = torch.tensor([row['event']], dtype=torch.long)

        # .npy 파일들 로드
        feature_files = glob.glob(os.path.join(feature_folder, "*.npy"))
        feature_files = [f for f in feature_files if "__MACOSX" not in f and "._" not in os.path.basename(f)]

        try:
            all_feature_vectors = [np.load(f) for f in feature_files]
            x = torch.tensor(np.stack(all_feature_vectors), dtype=torch.float)
        except Exception as e:
            print(f"Warning: Failed to load {case_id}/{feature_folder}: {e}")
            return Data(x=None, case_id=case_id)

        # k-NN 그래프 구성 (수동 구현 사용)
        x_device = x.to(self.device_for_knn)
        edge_index = manual_knn_graph(x_device, k=min(K_FOR_KNN, len(x)-1))
        x = x.to('cpu')
        edge_index = edge_index.to('cpu')

        return Data(x=x, edge_index=edge_index, time=time, event=event, case_id=case_id)

# ========================
# 4. GAT 모델 정의
# ========================

class GATSurvival(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=8, dropout=0.5):
        super(GATSurvival, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout
        
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        
        graph_vector = global_mean_pool(x, batch)
        risk_score = self.classifier(graph_vector)
        return risk_score

# ========================
# 5. Cox PH Loss
# ========================

class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()
    
    def forward(self, risk_scores, times, events):
        # squeeze 대신 flatten 사용 (배치 크기 1일 때도 1D 유지)
        risk_scores = risk_scores.view(-1)  # (batch,) 형태로 변환
        times = times.view(-1)
        events = events.view(-1).float()
        
        # 배치 크기가 1 이하면 loss 계산 불가
        if len(risk_scores) <= 1:
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        
        # 시간 순으로 정렬
        sorted_indices = torch.argsort(times, descending=True)
        risk_scores = risk_scores[sorted_indices]
        events = events[sorted_indices]
        
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-7)
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * events
        
        num_events = events.sum()
        if num_events > 0:
            loss = -censored_likelihood.sum() / num_events
        else:
            loss = torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        
        return loss

# ========================
# 6. 평가 함수
# ========================

@torch.no_grad()
def evaluate_survival(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_risk_scores, all_times, all_events = [], [], []
    
    for data in loader:
        if not hasattr(data, 'x') or data.x is None:
            continue
        data = data.to(device)
        risk_scores = model(data)
        loss = criterion(risk_scores, data.time, data.event)
        
        total_loss += loss.item()
        all_risk_scores.append(risk_scores.cpu())
        all_times.append(data.time.cpu())
        all_events.append(data.event.cpu())
    
    if not all_risk_scores:
        return 0, 0.5, (np.array([]), np.array([]), np.array([]))
    
    risk_scores_np = torch.cat(all_risk_scores).numpy().flatten()
    times_np = torch.cat(all_times).numpy().flatten()
    events_np = torch.cat(all_events).numpy().flatten().astype(bool)
    
    try:
        c_index = concordance_index(times_np, risk_scores_np, events_np)
    except ValueError:
        c_index = 0.5
    
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    
    return avg_loss, c_index, (risk_scores_np, times_np, events_np)

# ========================
# 7. 학습 함수
# ========================

def train_survival_model(train_loader, test_loader, model_name, input_dim=2048, num_epochs=50):
    """단일 모델 학습"""
    
    model = GATSurvival(input_dim=input_dim, hidden_dim=128, heads=8, dropout=0.5).to(device)
    criterion = CoxPHLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    
    best_c_index = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, f"best_model_{model_name}.pt")
    history = {'train_loss': [], 'test_loss': [], 'test_c_index': []}
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} Model")
    print(f"{'='*60}")
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for data in pbar:
            if not hasattr(data, 'x') or data.x is None:
                continue
            data = data.to(device)
            
            optimizer.zero_grad()
            risk_scores = model(data)
            loss = criterion(risk_scores, data.time, data.event)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= (len(train_loader) + 1e-6)
        
        # Validation
        test_loss, test_c_index, _ = evaluate_survival(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_c_index'].append(test_c_index)
        
        print(f"[Epoch {epoch:03d}/{num_epochs:03d}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f} | Test C-Index: {test_c_index:.4f}")
        
        # Best model 저장
        if test_c_index > best_c_index:
            best_c_index = test_c_index
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New Best Model Saved (C-Index: {best_c_index:.4f})")
        print("-" * 60)
    
    print(f"\nBest Test C-Index: {best_c_index:.4f}")
    
    return best_model_path, history, best_c_index

# ========================
# 8. 최종 평가 함수
# ========================

def plot_kaplan_meier(risk_scores, times, events, title, save_path):
    """Kaplan-Meier 생존곡선"""
    median_risk = np.median(risk_scores)
    high_risk = risk_scores > median_risk
    low_risk = risk_scores <= median_risk
    
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kmf.fit(times[high_risk], events[high_risk], label="High Risk")
    kmf.plot_survival_function(ax=ax, color='red', linewidth=2)
    
    kmf.fit(times[low_risk], events[low_risk], label="Low Risk")
    kmf.plot_survival_function(ax=ax, color='blue', linewidth=2)
    
    # Log-rank test
    results = logrank_test(times[high_risk], times[low_risk],
                          events[high_risk], events[low_risk])
    p_value = results.p_value
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.figtext(0.15, 0.15, f"Log-rank test p-value: {p_value:.4e}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return p_value

def final_evaluation(model, test_loader, train_loader, model_name):
    """최종 평가"""
    
    criterion = CoxPHLoss()
    
    # 평가
    test_loss, test_c_index, (test_risks, test_times, test_events) = \
        evaluate_survival(model, test_loader, criterion, device)
    _, _, (train_risks, train_times, train_events) = \
        evaluate_survival(model, train_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Final Test Results")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test C-Index: {test_c_index:.4f}")
    
    metrics = {
        'model_name': model_name,
        'test_loss': float(test_loss),
        'test_c_index': float(test_c_index),
        'num_test_samples': len(test_times),
        'num_events': int(test_events.sum())
    }
    
    # Time-dependent AUC
    print("\n--- Time-dependent AUC ---")
    time_points = np.linspace(min(test_times.min(), 365), min(test_times.max(), 365*5), 5)
    
    try:
        auc_scores = []
        for t in time_points:
            auc = time_dependent_auc(
                train_times, train_events, 
                test_times, test_events, 
                test_risks, t
            )
            if auc is not None:
                auc_scores.append(auc)
        
        if len(auc_scores) > 0:
            mean_auc = np.mean(auc_scores)
            print(f"Mean t-AUC: {mean_auc:.4f}")
            metrics['mean_t_auc'] = float(mean_auc)
            metrics['t_auc'] = {}
            
            for t, auc in zip(time_points[:len(auc_scores)], auc_scores):
                year = t / 365.25
                print(f"  t-AUC @ {year:.1f} years: {auc:.4f}")
                metrics['t_auc'][f'{year:.1f}_years'] = float(auc)
        else:
            print("Could not calculate t-AUC")
            metrics['mean_t_auc'] = None
    except Exception as e:
        print(f"t-AUC calculation failed: {e}")
        metrics['mean_t_auc'] = None
    
    # Kaplan-Meier Plot
    print("\n--- Kaplan-Meier Plot ---")
    km_plot_path = os.path.join(OUTPUT_DIR, f"km_plot_{model_name}.png")
    p_value = plot_kaplan_meier(test_risks, test_times, test_events,
                                f"Kaplan-Meier Curve ({model_name})",
                                km_plot_path)
    metrics['logrank_p_value'] = float(p_value)
    print(f"KM Plot saved: {km_plot_path}")
    print(f"Log-rank p-value: {p_value:.4e}")
    
    # 메트릭 저장
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{model_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved: {metrics_path}")
    
    return metrics

# ========================
# 9. 메인 실행 함수
# ========================

def main():
    print("=" * 70)
    print("FS vs FFPE Survival Prediction Comparison (Graph-based)")
    print("=" * 70)
    
    # 생존 데이터 로드
    survival_df = prepare_survival_data(TSV_PATH)
    
    # Feature 매칭
    print("\n--- Matching Features ---")
    fs_data = find_feature_folders(FS_FEATURE_DIR, survival_df)
    ffpe_data = find_feature_folders(FFPE_FEATURE_DIR, survival_df)
    
    print(f"FS samples: {len(fs_data)}")
    print(f"FFPE samples: {len(ffpe_data)}")
    
    # Paired 샘플 선택
    fs_cases = set(fs_data['case_id'].unique())
    ffpe_cases = set(ffpe_data['case_id'].unique())
    paired_cases = fs_cases & ffpe_cases
    
    print(f"\nPaired cases: {len(paired_cases)}")
    
    if len(paired_cases) == 0:
        print("ERROR: No paired cases found! Check feature directories.")
        return
    
    fs_paired = fs_data[fs_data['case_id'].isin(paired_cases)].reset_index(drop=True)
    ffpe_paired = ffpe_data[ffpe_data['case_id'].isin(paired_cases)].reset_index(drop=True)
    
    # Train/Test Split
    from sklearn.model_selection import train_test_split
    train_cases, test_cases = train_test_split(
        list(paired_cases), test_size=0.2, random_state=42
    )
    
    print(f"Train cases: {len(train_cases)}")
    print(f"Test cases: {len(test_cases)}")
    
    # 데이터셋 생성
    fs_train = fs_paired[fs_paired['case_id'].isin(train_cases)]
    fs_test = fs_paired[fs_paired['case_id'].isin(test_cases)]
    ffpe_train = ffpe_paired[ffpe_paired['case_id'].isin(train_cases)]
    ffpe_test = ffpe_paired[ffpe_paired['case_id'].isin(test_cases)]
    
    # PyG Dataset
    fs_train_dataset = WsiSurvivalGraphDataset(fs_train, device)
    fs_test_dataset = WsiSurvivalGraphDataset(fs_test, device)
    ffpe_train_dataset = WsiSurvivalGraphDataset(ffpe_train, device)
    ffpe_test_dataset = WsiSurvivalGraphDataset(ffpe_test, device)
    
    # DataLoader
    batch_size = 8
    fs_train_loader = DataLoader(fs_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    fs_test_loader = DataLoader(fs_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ffpe_train_loader = DataLoader(ffpe_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    ffpe_test_loader = DataLoader(ffpe_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # FS 모델 학습
    print("\n" + "=" * 70)
    print("Training FS Model")
    print("=" * 70)
    fs_model_path, fs_history, fs_best_c = train_survival_model(
        fs_train_loader, fs_test_loader, "FS", input_dim=2048, num_epochs=20
    )
    
    # FFPE 모델 학습
    print("\n" + "=" * 70)
    print("Training FFPE Model")
    print("=" * 70)
    ffpe_model_path, ffpe_history, ffpe_best_c = train_survival_model(
        ffpe_train_loader, ffpe_test_loader, "FFPE", input_dim=2048, num_epochs=20
    )
    
    # 최종 평가
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    # FS 모델
    fs_model = GATSurvival(input_dim=2048, hidden_dim=128).to(device)
    fs_model.load_state_dict(torch.load(fs_model_path))
    fs_metrics = final_evaluation(fs_model, fs_test_loader, fs_train_loader, "FS")
    
    # FFPE 모델
    ffpe_model = GATSurvival(input_dim=2048, hidden_dim=128).to(device)
    ffpe_model.load_state_dict(torch.load(ffpe_model_path))
    ffpe_metrics = final_evaluation(ffpe_model, ffpe_test_loader, ffpe_train_loader, "FFPE")
    
    # 비교 결과
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: FS vs FFPE")
    print("=" * 70)
    
    comparison = {
        'FS': {
            'C-Index': fs_metrics['test_c_index'],
            'Mean t-AUC': fs_metrics.get('mean_t_auc', 'N/A'),
            'Log-rank p-value': fs_metrics['logrank_p_value']
        },
        'FFPE': {
            'C-Index': ffpe_metrics['test_c_index'],
            'Mean t-AUC': ffpe_metrics.get('mean_t_auc', 'N/A'),
            'Log-rank p-value': ffpe_metrics['logrank_p_value']
        }
    }
    
    print("\nPerformance Comparison:")
    print(f"{'Metric':<20} {'FS':<15} {'FFPE':<15} {'Better':<10}")
    print("-" * 60)
    
    for metric in ['C-Index', 'Mean t-AUC', 'Log-rank p-value']:
        fs_val = comparison['FS'][metric]
        ffpe_val = comparison['FFPE'][metric]
        
        if metric == 'Log-rank p-value':
            winner = 'FS' if fs_val < ffpe_val else 'FFPE'
        else:
            if fs_val == 'N/A' or ffpe_val == 'N/A':
                winner = 'N/A'
            else:
                winner = 'FS' if fs_val > ffpe_val else 'FFPE'
        
        print(f"{metric:<20} {str(fs_val):<15} {str(ffpe_val):<15} {winner:<10}")
    
    # 저장
    comparison_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"\nAll results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()