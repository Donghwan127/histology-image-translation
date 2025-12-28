import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 부드러운 파스텔 색상 팔레트 (FFPE 강조용 그라데이션)
COLOR_FS = '#A8D5E2'      # 연한 블루 (FS)
COLOR_FFPE = '#6B9AC4'    # 진한 블루 (FFPE - 강조)
COLOR_IMPROVEMENT = '#81C784'  # 부드러운 그린 (개선)
COLOR_DECLINE = '#FFB74D'      # 부드러운 오렌지 (하락)
COLOR_SIGNIFICANT = '#66BB6A'   # 그린 (유의미)
COLOR_NOT_SIGNIFICANT = '#EF9A9A'  # 핑크 (비유의미)

# 결과 데이터
results = {
    'FS': {
        'C-Index': 0.5320,
        'Mean t-AUC': 0.6721,
        't-AUC @ 0.7yr': 0.7069,
        't-AUC @ 1.7yr': 0.4905,
        't-AUC @ 2.6yr': 0.6933,
        't-AUC @ 3.6yr': 0.7976,
        'Log-rank p-value': 0.4538
    },
    'FFPE': {
        'C-Index': 0.6126,
        'Mean t-AUC': 0.8128,
        't-AUC @ 0.7yr': 0.8103,
        't-AUC @ 1.7yr': 0.6810,
        't-AUC @ 2.6yr': 0.9267,
        't-AUC @ 3.6yr': 0.8333,
        'Log-rank p-value': 0.0446
    }
}

# ========================
# 종합 대시보드 (개선된 디자인)
# ========================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ========================
# Subplot 1: 전체 성능 비교
# ========================
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['C-Index', 'Mean t-AUC']
x = np.arange(len(metrics))
width = 0.35
fs_values = [results['FS'][m] for m in metrics]
ffpe_values = [results['FFPE'][m] for m in metrics]

bars1 = ax1.bar(x - width/2, fs_values, width, label='FS', 
               color=COLOR_FS, edgecolor='white', linewidth=2)
bars2 = ax1.bar(x + width/2, ffpe_values, width, label='FFPE', 
               color=COLOR_FFPE, edgecolor='white', linewidth=2)

# 값 표시 (막대 위 중앙에 배치)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='600')

ax1.set_ylabel('Score', fontsize=13, fontweight='600')
ax1.set_title('(A) Overall Performance', fontsize=14, fontweight='700', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=12, fontweight='500')
ax1.legend(fontsize=11, frameon=False, loc='upper left')
ax1.set_ylim([0, 1])
ax1.axhline(y=0.5, color='#BDBDBD', linestyle='--', alpha=0.6, linewidth=1.5, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#BDBDBD')
ax1.spines['bottom'].set_color('#BDBDBD')
ax1.tick_params(colors='#424242')

# ========================
# Subplot 2: 시간별 AUC
# ========================
ax2 = fig.add_subplot(gs[0, 1])
time_points = [0.7, 1.7, 2.6, 3.6]
fs_aucs = [results['FS'][f't-AUC @ {t}yr'] for t in time_points]
ffpe_aucs = [results['FFPE'][f't-AUC @ {t}yr'] for t in time_points]

# FS 라인 (연한 색)
ax2.plot(time_points, fs_aucs, 'o-', label='FS', linewidth=3, markersize=9, 
        color=COLOR_FS, markeredgecolor='white', markeredgewidth=2.5, zorder=3)

# FFPE 라인 (진한 색, 강조)
ax2.plot(time_points, ffpe_aucs, 's-', label='FFPE', linewidth=3, markersize=10, 
        color=COLOR_FFPE, markeredgecolor='white', markeredgewidth=2.5, zorder=4)

# 값 표시 (그래프 선 끝에 안 걸치게 여백 추가)
for i, (t, fs_auc, ffpe_auc) in enumerate(zip(time_points, fs_aucs, ffpe_aucs)):
    # FS 값
    offset_fs = 0.06 if fs_auc < ffpe_auc else -0.06
    ax2.text(t, fs_auc + offset_fs, f'{fs_auc:.2f}', ha='center', fontsize=10, 
            color=COLOR_FS, fontweight='600', bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', edgecolor='none', alpha=0.8))
    
    # FFPE 값
    offset_ffpe = -0.06 if fs_auc < ffpe_auc else 0.06
    ax2.text(t, ffpe_auc + offset_ffpe, f'{ffpe_auc:.2f}', ha='center', fontsize=10, 
            color=COLOR_FFPE, fontweight='600', bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', edgecolor='none', alpha=0.8))

# 랜덤 기준선
ax2.axhline(y=0.5, color='#BDBDBD', linestyle='--', alpha=0.6, linewidth=1.5, zorder=0)

ax2.set_xlabel('Time (Years)', fontsize=13, fontweight='600')
ax2.set_ylabel('Time-dependent AUC', fontsize=13, fontweight='600')
ax2.set_title('(B) Temporal Performance', fontsize=14, fontweight='700', pad=15)
ax2.legend(fontsize=11, frameon=False, loc='lower right')
ax2.set_ylim([0.35, 1.0])
ax2.set_xlim([0.5, 3.8])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#BDBDBD')
ax2.spines['bottom'].set_color('#BDBDBD')
ax2.tick_params(colors='#424242')

# ========================
# Subplot 3: 개선율
# ========================
ax3 = fig.add_subplot(gs[1, 0])
metrics_labels = ['C-Index', 't-AUC\n0.7yr', 't-AUC\n1.7yr', 't-AUC\n2.6yr', 't-AUC\n3.6yr', 'Mean\nt-AUC']
fs_vals = [results['FS']['C-Index'], results['FS']['t-AUC @ 0.7yr'], 
           results['FS']['t-AUC @ 1.7yr'], results['FS']['t-AUC @ 2.6yr'],
           results['FS']['t-AUC @ 3.6yr'], results['FS']['Mean t-AUC']]
ffpe_vals = [results['FFPE']['C-Index'], results['FFPE']['t-AUC @ 0.7yr'],
             results['FFPE']['t-AUC @ 1.7yr'], results['FFPE']['t-AUC @ 2.6yr'],
             results['FFPE']['t-AUC @ 3.6yr'], results['FFPE']['Mean t-AUC']]
improvements = [(ffpe - fs) / fs * 100 for fs, ffpe in zip(fs_vals, ffpe_vals)]

# 색상 (개선=그린, 하락=오렌지)
colors = [COLOR_IMPROVEMENT if imp > 0 else COLOR_DECLINE for imp in improvements]
bars = ax3.bar(range(len(metrics_labels)), improvements, color=colors, 
              edgecolor='white', linewidth=2)

# 값 표시 (막대 끝 안쪽에 배치)
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    y_pos = height + (2 if height > 0 else -2)
    va = 'bottom' if height > 0 else 'top'
    ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
            ha='center', va=va, fontsize=11, fontweight='600',
            color='#424242')

ax3.set_xticks(range(len(metrics_labels)))
ax3.set_xticklabels(metrics_labels, fontsize=10, fontweight='500')
ax3.set_ylabel('Improvement (%)', fontsize=13, fontweight='600')
ax3.set_title('(C) FFPE Improvement over FS', fontsize=14, fontweight='700', pad=15)
ax3.axhline(y=0, color='#757575', linestyle='-', linewidth=1.5, zorder=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color('#BDBDBD')
ax3.spines['bottom'].set_color('#BDBDBD')
ax3.tick_params(colors='#424242')
ax3.set_ylim([-5, 45])

# ========================
# Subplot 4: 통계적 유의성
# ========================
ax4 = fig.add_subplot(gs[1, 1])
p_values = [results['FS']['Log-rank p-value'], results['FFPE']['Log-rank p-value']]
labels = ['FS', 'FFPE']
colors_sig = [COLOR_NOT_SIGNIFICANT, COLOR_SIGNIFICANT]

bars = ax4.bar(labels, p_values, color=colors_sig, edgecolor='white', linewidth=2, width=0.5)

# p=0.05 기준선
ax4.axhline(y=0.05, color='#E57373', linestyle='--', linewidth=2, 
           label='Significance threshold\n(p = 0.05)', zorder=0)

# 값 표시 (막대 위에 깔끔하게)
for bar, p_val, label in zip(bars, p_values, labels):
    height = bar.get_height()
    significance = '✓ Significant' if p_val < 0.05 else '✗ Not Significant'
    color = '#2E7D32' if p_val < 0.05 else '#C62828'
    
    # p-value 표시
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.025,
            f'p = {p_val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='600',
            color='#424242')
    
    # 유의성 표시
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.065,
            significance,
            ha='center', va='bottom', fontsize=10, fontweight='600',
            color=color)

ax4.set_ylabel('p-value', fontsize=13, fontweight='600')
ax4.set_title('(D) Statistical Significance (Log-rank Test)', fontsize=14, fontweight='700', pad=15)
ax4.set_ylim([0, 0.55])
ax4.legend(fontsize=10, frameon=False, loc='upper right')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color('#BDBDBD')
ax4.spines['bottom'].set_color('#BDBDBD')
ax4.tick_params(colors='#424242')

# 전체 제목
plt.suptitle('FS vs FFPE: Comprehensive Performance Analysis', 
             fontsize=18, fontweight='700', y=0.98, color='#212121')

plt.savefig('fig_comprehensive_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: fig_comprehensive_clean.png")
plt.close()

# ========================
# 개별 그래프들도 깔끔하게 재생성
# ========================

# Figure 1: 전체 성능 비교 (단독)
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['C-Index', 'Mean t-AUC']
x = np.arange(len(metrics))
width = 0.35
fs_values = [results['FS'][m] for m in metrics]
ffpe_values = [results['FFPE'][m] for m in metrics]

bars1 = ax.bar(x - width/2, fs_values, width, label='FS', 
              color=COLOR_FS, edgecolor='white', linewidth=2.5)
bars2 = ax.bar(x + width/2, ffpe_values, width, label='FFPE', 
              color=COLOR_FFPE, edgecolor='white', linewidth=2.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', 
                fontsize=13, fontweight='600')

ax.set_ylabel('Score', fontsize=14, fontweight='600')
ax.set_title('Overall Performance Comparison', fontsize=16, fontweight='700', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13, fontweight='500')
ax.legend(fontsize=12, frameon=False, loc='upper left')
ax.set_ylim([0, 1])
ax.axhline(y=0.5, color='#BDBDBD', linestyle='--', alpha=0.6, linewidth=2, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDBDBD')
ax.spines['bottom'].set_color('#BDBDBD')
ax.tick_params(colors='#424242')

plt.tight_layout()
plt.savefig('fig1_overall_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: fig1_overall_clean.png")
plt.close()

# Figure 2: 시간별 AUC (단독)
fig, ax = plt.subplots(figsize=(11, 7))
time_points = [0.7, 1.7, 2.6, 3.6]
fs_aucs = [results['FS'][f't-AUC @ {t}yr'] for t in time_points]
ffpe_aucs = [results['FFPE'][f't-AUC @ {t}yr'] for t in time_points]

ax.plot(time_points, fs_aucs, 'o-', label='FS', linewidth=4, markersize=12, 
       color=COLOR_FS, markeredgecolor='white', markeredgewidth=3, zorder=3)
ax.plot(time_points, ffpe_aucs, 's-', label='FFPE', linewidth=4, markersize=13, 
       color=COLOR_FFPE, markeredgecolor='white', markeredgewidth=3, zorder=4)

# 값 표시
for i, (t, fs_auc, ffpe_auc) in enumerate(zip(time_points, fs_aucs, ffpe_aucs)):
    offset_fs = 0.06 if fs_auc < ffpe_auc else -0.06
    ax.text(t, fs_auc + offset_fs, f'{fs_auc:.3f}', ha='center', fontsize=11, 
           color=COLOR_FS, fontweight='600', bbox=dict(boxstyle='round,pad=0.4', 
           facecolor='white', edgecolor='none', alpha=0.9))
    
    offset_ffpe = -0.06 if fs_auc < ffpe_auc else 0.06
    ax.text(t, ffpe_auc + offset_ffpe, f'{ffpe_auc:.3f}', ha='center', fontsize=11, 
           color=COLOR_FFPE, fontweight='600', bbox=dict(boxstyle='round,pad=0.4', 
           facecolor='white', edgecolor='none', alpha=0.9))

ax.axhline(y=0.5, color='#BDBDBD', linestyle='--', alpha=0.6, linewidth=2, 
          label='Random (0.5)', zorder=0)

ax.set_xlabel('Time (Years)', fontsize=14, fontweight='600')
ax.set_ylabel('Time-dependent AUC', fontsize=14, fontweight='600')
ax.set_title('Temporal Performance: Time-dependent AUC Over Time', 
            fontsize=16, fontweight='700', pad=20)
ax.legend(fontsize=12, frameon=False, loc='lower right')
ax.set_ylim([0.35, 1.0])
ax.set_xlim([0.5, 3.8])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDBDBD')
ax.spines['bottom'].set_color('#BDBDBD')
ax.tick_params(colors='#424242', labelsize=11)

plt.tight_layout()
plt.savefig('fig2_temporal_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: fig2_temporal_clean.png")
plt.close()

# Figure 3: 개선율 (단독)
fig, ax = plt.subplots(figsize=(12, 7))
metrics_labels = ['C-Index', 't-AUC @ 0.7yr', 't-AUC @ 1.7yr', 
                  't-AUC @ 2.6yr', 't-AUC @ 3.6yr', 'Mean t-AUC']
improvements_clean = improvements.copy()

colors = [COLOR_IMPROVEMENT if imp > 0 else COLOR_DECLINE for imp in improvements_clean]
bars = ax.bar(range(len(metrics_labels)), improvements_clean, color=colors, 
             edgecolor='white', linewidth=2.5, width=0.7)

for bar, imp in zip(bars, improvements_clean):
    height = bar.get_height()
    y_pos = height + (2 if height > 0 else -2)
    va = 'bottom' if height > 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
           f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
           ha='center', va=va, fontsize=12, fontweight='600',
           color='#424242')

ax.set_xticks(range(len(metrics_labels)))
ax.set_xticklabels(metrics_labels, fontsize=12, fontweight='500', rotation=0)
ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='600')
ax.set_title('FFPE Performance Improvement over FS', 
            fontsize=16, fontweight='700', pad=20)
ax.axhline(y=0, color='#757575', linestyle='-', linewidth=2, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDBDBD')
ax.spines['bottom'].set_color('#BDBDBD')
ax.tick_params(colors='#424242', labelsize=11)
ax.set_ylim([-5, 45])

plt.tight_layout()
plt.savefig('fig3_improvement_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: fig3_improvement_clean.png")
plt.close()

# Figure 4: 통계적 유의성 (단독)
fig, ax = plt.subplots(figsize=(9, 7))
p_values = [results['FS']['Log-rank p-value'], results['FFPE']['Log-rank p-value']]
labels = ['FS', 'FFPE']
colors_sig = [COLOR_NOT_SIGNIFICANT, COLOR_SIGNIFICANT]

bars = ax.bar(labels, p_values, color=colors_sig, edgecolor='white', linewidth=3, width=0.5)
ax.axhline(y=0.05, color='#E57373', linestyle='--', linewidth=2.5, 
          label='Significance threshold (p = 0.05)', zorder=0)

for bar, p_val in zip(bars, p_values):
    height = bar.get_height()
    significance = '✓ Significant' if p_val < 0.05 else '✗ Not Significant'
    color = '#2E7D32' if p_val < 0.05 else '#C62828'
    
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
           f'p = {p_val:.4f}',
           ha='center', va='bottom', fontsize=12, fontweight='600',
           color='#424242')
    
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.08,
           significance,
           ha='center', va='bottom', fontsize=11, fontweight='600',
           color=color)

ax.set_ylabel('p-value', fontsize=14, fontweight='600')
ax.set_title('Statistical Significance: Log-rank Test', 
            fontsize=16, fontweight='700', pad=20)
ax.set_ylim([0, 0.6])
ax.legend(fontsize=11, frameon=False, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDBDBD')
ax.spines['bottom'].set_color('#BDBDBD')
ax.tick_params(colors='#424242', labelsize=11)

plt.tight_layout()
plt.savefig('fig4_significance_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: fig4_significance_clean.png")
plt.close()

print("\n" + "="*70)
print("✅ All clean figures saved successfully!")
print("="*70)
print("\nGenerated files:")
print("  📊 fig_comprehensive_clean.png  (논문용 추천 - 4개 subplot)")
print("  📊 fig1_overall_clean.png       (전체 성능)")
print("  📊 fig2_temporal_clean.png      (시간별 AUC - 핵심!)")
print("  📊 fig3_improvement_clean.png   (개선율)")
print("  📊 fig4_significance_clean.png  (통계적 유의성)")
print("="*70)