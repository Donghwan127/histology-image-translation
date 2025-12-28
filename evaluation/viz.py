import matplotlib.pyplot as plt
import numpy as np

# 폰트 및 스타일 설정 (서버 환경에 Arial이 없을 경우를 대비해 sans-serif 설정 포함)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트
COLOR_REAL_FF = '#A8D5E2'      # 연한 블루
COLOR_FAKE_FFPE = '#FFB74D'    # 오렌지
COLOR_REAL_FFPE = '#6B9AC4'    # 진한 블루

# 데이터 설정 (제공해주신 데이터와 동일)
results = {
    'Real FF': {'C-Index': 0.7917, 'Mean t-AUC': 0.8826},
    'Fake FFPE': {'C-Index': 0.8004, 'Mean t-AUC': 0.8900},
    'Real FFPE': {'C-Index': 0.8466, 'Mean t-AUC': 0.9264}
}

pairwise_stats = [
    ['Real FF vs Fake FFPE', '+0.0087', '0.4929', 'Not Significant'],
    ['Fake FFPE vs Real FFPE', '+0.0462', '0.0058', 'Significant **'],
    ['Real FF vs Real FFPE', '+0.0549', '0.0042', 'Significant **']
]

# 개선율 계산 로직
def get_imp(val, base): return (val - base) / base * 100
fake_imp = [get_imp(results['Fake FFPE']['C-Index'], results['Real FF']['C-Index']),
            get_imp(results['Fake FFPE']['Mean t-AUC'], results['Real FF']['Mean t-AUC'])]
real_imp = [get_imp(results['Real FFPE']['C-Index'], results['Real FF']['C-Index']),
            get_imp(results['Real FFPE']['Mean t-AUC'], results['Real FF']['Mean t-AUC'])]

# ========================
# Dashboard Layout (2 rows, 2 cols)
# ========================
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# ---------------------------------------------------------
# Subplot A: Overall Performance Comparison
# ---------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['C-Index', 'Mean t-AUC']
x = np.arange(len(metrics))
width = 0.25
models = ['Real FF', 'Fake FFPE', 'Real FFPE']
colors = [COLOR_REAL_FF, COLOR_FAKE_FFPE, COLOR_REAL_FFPE]

for i, (model, color) in enumerate(zip(models, colors)):
    vals = [results[model][m] for m in metrics]
    bars = ax1.bar(x + (i - 1) * width, vals, width, label=model, color=color, edgecolor='white', linewidth=1.5)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_title('(A) Overall Performance Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x); ax1.set_xticklabels(metrics)
ax1.set_ylim(0.6, 1.0)
ax1.legend(frameon=False); ax1.spines[['top', 'right']].set_visible(False)

# ---------------------------------------------------------
# Subplot B: Performance Improvement over Real FF (Overlapping Fixed)
# ---------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
x_imp = np.arange(len(metrics))
width_imp = 0.35

bars_f = ax2.bar(x_imp - width_imp/2, fake_imp, width_imp, label='Fake FFPE vs Real FF', color=COLOR_FAKE_FFPE, edgecolor='white')
bars_r = ax2.bar(x_imp + width_imp/2, real_imp, width_imp, label='Real FFPE vs Real FF', color=COLOR_REAL_FFPE, edgecolor='white')

# 텍스트 겹침 해결: y_pos 간격을 더 확보하고 폰트 조정
for bars in [bars_f, bars_r]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                 f'+{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

ax2.set_title('(B) Performance Improvement over Real FF', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_imp); ax2.set_xticklabels(metrics)
ax2.set_ylabel('Improvement (%)', fontweight='bold')
ax2.set_ylim(0, 8.5) # 텍스트 공간 확보를 위해 상단 여유 증가
ax2.legend(frameon=False); ax2.spines[['top', 'right']].set_visible(False)

# ---------------------------------------------------------
# Subplot C: Pairwise Statistical Summary Table (D 대체)
# ---------------------------------------------------------
ax3 = fig.add_subplot(gs[1, :]) # 하단 전체 너비 사용
ax3.axis('off')

columns = ['Comparison Pair', 'C-Index Delta (Δ)', 'p-value (t-test)', 'Significance State']
table = ax3.table(cellText=pairwise_stats, colLabels=columns, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 4.0) # 행 높이 조절

# 테이블 스타일링
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#455A64')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        # Significance에 따른 색상 강조
        if col == 3 and 'Significant' in cell.get_text().get_text() and '*' in cell.get_text().get_text():
            cell.set_text_props(color='#2E7D32', fontweight='bold') # Green
        elif col == 3:
            cell.set_text_props(color='#C62828') # Red

ax3.set_title('(C) Pairwise Statistical Comparison Summary', fontsize=14, fontweight='bold', y=0.85)

# 메인 제목
plt.suptitle('Survival Prediction Analysis: Real FF vs Fake FFPE vs Real FFPE\n(Comprehensive 3-Way Comparison)', 
             fontsize=17, fontweight='bold', y=0.98)

plt.savefig('survival_analysis_final.png', dpi=300, bbox_inches='tight')
print("✅ 최종 결과물이 'survival_analysis_final.png'로 저장되었습니다.")
plt.show()