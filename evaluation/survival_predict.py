# survival_predict_extended.py
# 
# 변경사항:
# - Epoch: 30 → 100
# - 5번 반복 실험으로 평균±표준편차
# - 더 명확한 시각화

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

# ========================
# 설정
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TSV_PATH = "LUAD_clinical.tsv"
REAL_FF_FEATURE_DIR = "features/real_ff"
FAKE_FFPE_FEATURE_DIR = "features/fake_ffpe"
REAL_FFPE_FEATURE_DIR = "features/real_ffpe"
OUTPUT_DIR = "survival_comparison_extended"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 핵심 설정
K_FOR_KNN = 8
NUM_EPOCHS = 100  # 늘림
NUM_RUNS = 5      # 5번 반복

# ========================
# 유틸리티 함수들
# ========================

def manual_knn_graph(x, k):
    num_nodes = x.size(0)
    if k <= 0 or num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=x.device)
    k = min(k, num_nodes - 1)
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    dist = torch.sum(diff ** 2, dim=2)
    _, indices = torch.topk(dist, k=k+1, largest=False, dim=1)
    indices = indices[:, 1:k+1]
    src_nodes = torch.arange(num_nodes, device=x.device).unsqueeze(1).repeat(1, k)
    edge_index = torch.stack([src_nodes.flatten(), indices.flatten()], dim=0)
    return edge_index

def concordance_index(event_times, predicted_scores, event_observed):
    n = len(event_times)
    if n == 0:
        return 0.5
    concordant, discordant, tied = 0, 0, 0
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
                    tied += 1
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5

def time_dependent_auc(times, events, risks, time_point):
    case_idx = (times <= time_point) & events
    control_idx = times > time_point
    if np.sum(case_idx) == 0 or np.sum(control_idx) == 0:
        return None
    case_scores = risks[case_idx]
    control_scores = risks[control_idx]
    concordant, total = 0, 0
    for cs in case_scores:
        for cts in control_scores:
            if cs > cts:
                concordant += 1
            elif cs == cts:
                concordant += 0.5
            total += 1
    return concordant / total if total > 0 else None

# ========================
# 데이터 로딩
# ========================

def prepare_survival_data(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    survival_df = df.groupby('case_submitter_id').agg({
        'days_to_death': 'first',
        'days_to_last_follow_up': 'first',
        'vital_status': 'first',
    }).reset_index()
    
    survival_df['time'] = survival_df.apply(
        lambda row: row['days_to_death'] if pd.notna(row['days_to_death']) and row['days_to_death'] != '--'
        else row['days_to_last_follow_up'] if pd.notna(row['days_to_last_follow_up']) and row['days_to_last_follow_up'] != '--'
        else np.nan, axis=1
    )
    survival_df['event'] = survival_df['vital_status'].apply(lambda x: 1 if x == 'Dead' else 0)
    survival_df = survival_df[survival_df['time'].notna()].copy()
    survival_df['time'] = pd.to_numeric(survival_df['time'], errors='coerce')
    survival_df = survival_df[survival_df['time'] > 0].copy()
    
    print(f"Patients: {len(survival_df)}, Deaths: {survival_df['event'].sum()}, Censored: {(1-survival_df['event']).sum()}")
    return survival_df

def find_feature_folders(feature_root_dir, survival_df):
    matched = []
    if not os.path.exists(feature_root_dir):
        return pd.DataFrame(matched)
    
    for pid in survival_df['case_submitter_id'].unique():
        pdir = os.path.join(feature_root_dir, pid)
        if not os.path.exists(pdir):
            continue
        for folder in os.listdir(pdir):
            fpath = os.path.join(pdir, folder)
            if os.path.isdir(fpath):
                npys = [f for f in glob.glob(os.path.join(fpath, "*.npy")) 
                       if "MACOSX" not in f and not os.path.basename(f).startswith("._")]
                if npys:
                    row = survival_df[survival_df['case_submitter_id'] == pid].iloc[0]
                    matched.append({
                        'case_id': pid, 'feature_folder': fpath,
                        'num_patches': len(npys), 'time': row['time'], 'event': row['event']
                    })
    return pd.DataFrame(matched)

# ========================
# Dataset & Model
# ========================

class WsiDataset(Dataset):
    def __init__(self, data_df, dev):
        self.data = data_df.reset_index(drop=True)
        self.dev = dev
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        files = [f for f in glob.glob(os.path.join(row['feature_folder'], "*.npy"))
                if "MACOSX" not in f and not os.path.basename(f).startswith("._")]
        if not files:
            return Data(x=None, case_id=row['case_id'])
        try:
            feats = [np.load(f) for f in files]
            x = torch.tensor(np.stack(feats), dtype=torch.float)
        except:
            return Data(x=None, case_id=row['case_id'])
        
        k = min(K_FOR_KNN, len(x)-1)
        if k > 0:
            edge_index = manual_knn_graph(x.to(self.dev), k).cpu()
        else:
            edge_index = torch.zeros((2,0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index,
                   time=torch.tensor([row['time']], dtype=torch.float),
                   event=torch.tensor([row['event']], dtype=torch.long),
                   case_id=row['case_id'])

class GATSurvival(nn.Module):
    def __init__(self, in_dim=2048, hid=128, heads=8, drop=0.5):
        super().__init__()
        self.drop = drop
        self.gat1 = GATConv(in_dim, hid, heads=heads, dropout=drop)
        self.gat2 = GATConv(hid*heads, hid, heads=1, dropout=drop)
        self.fc = nn.Sequential(
            nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid//2, 1)
        )
    
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, self.drop, self.training)
        x = F.elu(self.gat1(x, ei))
        x = F.dropout(x, self.drop, self.training)
        x = F.elu(self.gat2(x, ei))
        return self.fc(global_mean_pool(x, batch))

class CoxLoss(nn.Module):
    def forward(self, risk, time, event):
        risk, time, event = risk.view(-1), time.view(-1), event.view(-1).float()
        if len(risk) <= 1 or event.sum() == 0:
            return torch.tensor(0.0, device=risk.device, requires_grad=True)
        order = torch.argsort(time, descending=True)
        risk, event = risk[order], event[order]
        haz = torch.exp(risk)
        log_cumsum = torch.log(torch.cumsum(haz, 0) + 1e-7)
        return -(risk - log_cumsum).mul(event).sum() / event.sum()

# ========================
# 학습 함수
# ========================

def train_single_run(loader, name, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = GATSurvival().to(device)
    criterion = CoxLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    
    best_c = 0.0
    best_state = None
    
    for epoch in range(1, epochs+1):
        model.train()
        for data in loader:
            if data.x is None:
                continue
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data.time, data.event)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        risks, times, events = [], [], []
        with torch.no_grad():
            for data in loader:
                if data.x is None:
                    continue
                data = data.to(device)
                r = model(data).cpu().numpy().flatten()
                risks.extend(r)
                times.extend(data.time.cpu().numpy().flatten())
                events.extend(data.event.cpu().numpy().flatten())
        
        c = concordance_index(np.array(times), np.array(risks), np.array(events).astype(bool))
        if c > best_c:
            best_c = c
            best_state = model.state_dict().copy()
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for data in loader:
            if data.x is None:
                continue
            data = data.to(device)
            risks.extend(model(data).cpu().numpy().flatten())
            times.extend(data.time.cpu().numpy().flatten())
            events.extend(data.event.cpu().numpy().flatten())
    
    risks, times, events = np.array(risks), np.array(times), np.array(events).astype(bool)
    c_index = concordance_index(times, risks, events)
    
    # t-AUC
    t_aucs = {}
    for t_year in [0.7, 1.7, 2.5, 3.5]:
        auc = time_dependent_auc(times, events, risks, t_year * 365)
        if auc:
            t_aucs[t_year] = auc
    mean_t_auc = np.mean(list(t_aucs.values())) if t_aucs else 0.5
    
    # p-value
    med = np.median(risks)
    high, low = risks > med, risks <= med
    try:
        p = logrank_test(times[high], times[low], events[high], events[low]).p_value
    except:
        p = 1.0
    
    return {'c_index': c_index, 'mean_t_auc': mean_t_auc, 't_aucs': t_aucs, 'p_value': p}

# ========================
# 메인
# ========================

def main():
    print("="*70)
    print("Extended Training (100 epochs x 5 runs)")
    print("="*70)
    
    survival_df = prepare_survival_data(TSV_PATH)
    
    print("\n--- Loading Features ---")
    data_dict = {
        'Real_FF': find_feature_folders(REAL_FF_FEATURE_DIR, survival_df),
        'Fake_FFPE': find_feature_folders(FAKE_FFPE_FEATURE_DIR, survival_df),
        'Real_FFPE': find_feature_folders(REAL_FFPE_FEATURE_DIR, survival_df)
    }
    
    for n, d in data_dict.items():
        print(f"  {n}: {len(d)} samples")
    
    # Common patients
    common = set.intersection(*[set(d['case_id'].unique()) for d in data_dict.values() if len(d) > 0])
    print(f"\nCommon patients: {len(common)}")
    
    if len(common) < 20:
        print("Not enough patients!")
        return
    
    # Filter
    for n in data_dict:
        data_dict[n] = data_dict[n][data_dict[n]['case_id'].isin(common)].reset_index(drop=True)
    
    # Create datasets and loaders
    datasets = {n: WsiDataset(d, device) for n, d in data_dict.items()}
    loaders = {n: DataLoader(ds, batch_size=8, shuffle=True, num_workers=0) for n, ds in datasets.items()}
    
    # Multiple runs
    all_results = {n: [] for n in data_dict.keys()}
    
    for run in range(NUM_RUNS):
        print(f"\n{'='*50}")
        print(f"Run {run+1}/{NUM_RUNS}")
        print('='*50)
        
        for name in ['Real_FF', 'Fake_FFPE', 'Real_FFPE']:
            result = train_single_run(loaders[name], name, NUM_EPOCHS, seed=run*100)
            all_results[name].append(result)
            print(f"  {name}: C-Index={result['c_index']:.4f}, t-AUC={result['mean_t_auc']:.4f}")
    
    # Aggregate results
    print("\n" + "="*70)
    print("FINAL RESULTS (Mean ± Std over 5 runs)")
    print("="*70)
    
    summary = {}
    for name in ['Real_FF', 'Fake_FFPE', 'Real_FFPE']:
        c_vals = [r['c_index'] for r in all_results[name]]
        t_vals = [r['mean_t_auc'] for r in all_results[name]]
        p_vals = [r['p_value'] for r in all_results[name]]
        
        summary[name] = {
            'c_index_mean': np.mean(c_vals),
            'c_index_std': np.std(c_vals),
            'c_indices': c_vals,
            't_auc_mean': np.mean(t_vals),
            't_auc_std': np.std(t_vals),
            'p_value_median': np.median(p_vals)
        }
    
    print(f"\n{'Model':<12} {'C-Index':<20} {'Mean t-AUC':<20} {'p-value (med)':<15}")
    print("-"*70)
    for name in ['Real_FF', 'Fake_FFPE', 'Real_FFPE']:
        s = summary[name]
        print(f"{name:<12} {s['c_index_mean']:.4f} ± {s['c_index_std']:.4f}    "
              f"{s['t_auc_mean']:.4f} ± {s['t_auc_std']:.4f}    {s['p_value_median']:.2e}")
    
    # Visualizations
    print("\n--- Generating Visualizations ---")
    
    # 1. C-Index comparison with error bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # (A) C-Index
    ax = axes[0]
    names = ['Real_FF', 'Fake_FFPE', 'Real_FFPE']
    means = [summary[n]['c_index_mean'] for n in names]
    stds = [summary[n]['c_index_std'] for n in names]
    colors = ['#87CEEB', '#FFA500', '#4682B4']
    
    bars = ax.bar(names, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Random')
    ax.set_ylabel('C-Index', fontsize=12)
    ax.set_title('(A) C-Index Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 0.8)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.02, f'{m:.3f}', 
               ha='center', fontsize=11, fontweight='bold')
    ax.legend()
    
    # (B) Mean t-AUC
    ax = axes[1]
    means = [summary[n]['t_auc_mean'] for n in names]
    stds = [summary[n]['t_auc_std'] for n in names]
    
    bars = ax.bar(names, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Random')
    ax.set_ylabel('Mean t-AUC', fontsize=12)
    ax.set_title('(B) Mean t-AUC Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 0.9)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.02, f'{m:.3f}', 
               ha='center', fontsize=11, fontweight='bold')
    ax.legend()
    
    # (C) Box plot of C-Index across runs
    ax = axes[2]
    data_box = [summary[n]['c_indices'] for n in names]
    bp = ax.boxplot(data_box, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)
    ax.set_ylabel('C-Index', fontsize=12)
    ax.set_title('(C) C-Index Distribution (5 runs)', fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_extended.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/comparison_extended.png")
    
    # 2. Pairwise comparison
    print("\n--- Pairwise Comparison (paired t-test) ---")
    from scipy import stats
    
    pairs = [('Real_FF', 'Fake_FFPE'), ('Fake_FFPE', 'Real_FFPE'), ('Real_FF', 'Real_FFPE')]
    for a, b in pairs:
        ca = summary[a]['c_indices']
        cb = summary[b]['c_indices']
        t_stat, p_val = stats.ttest_rel(ca, cb)
        diff = np.mean(cb) - np.mean(ca)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {a} vs {b}: diff={diff:+.4f}, p={p_val:.4f} {sig}")
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        # Convert numpy types to Python types
        save_summary = {}
        for n, s in summary.items():
            save_summary[n] = {
                'c_index_mean': float(s['c_index_mean']),
                'c_index_std': float(s['c_index_std']),
                'c_indices': [float(x) for x in s['c_indices']],
                't_auc_mean': float(s['t_auc_mean']),
                't_auc_std': float(s['t_auc_std']),
                'p_value_median': float(s['p_value_median'])
            }
        json.dump(save_summary, f, indent=4)
    
    print(f"\nAll results saved in: {OUTPUT_DIR}")
    
    # Final ranking
    print("\n--- Final Ranking ---")
    ranking = sorted(summary.items(), key=lambda x: x[1]['c_index_mean'])
    for i, (n, s) in enumerate(ranking, 1):
        print(f"  {i}. {n}: {s['c_index_mean']:.4f} ± {s['c_index_std']:.4f}")
    
    expected = ['Real_FF', 'Fake_FFPE', 'Real_FFPE']
    actual = [n for n, _ in ranking]
    
    if actual == expected:
        print("\n✅ Results match expected order: Real_FF < Fake_FFPE < Real_FFPE")
    else:
        print(f"\n⚠️  Actual: {' < '.join(actual)}")
        print(f"   Expected: {' < '.join(expected)}")

if __name__ == "__main__":
    main()