import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid", palette="husl")
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'axes.facecolor': '#f8f8f8', 'figure.facecolor': '#ffffff',
    'grid.color': '#ffffff', 'grid.alpha': 0.4
})

BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "outputs" / "plots" / "ephemeral"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    'nafdpm': '#FF6B6B', 'sbb': '#4ECDC4', 'docentr': '#45B7D1', 'dplinknet': '#96CEB4', 'fdnet': '#FFBE0B',
    'background': '#ffffff', 'text': '#000000', 'text_inv': '#ffffff', 'quality': '#4682B4', 'speed': '#DC143C',
    'header': '#2C3E50', 'cell': '#ECF0F1', 'best': '#2ECC71', 'runtime': '#F39C12', 'border': '#34495e'
}

# we needed normalization due to different scales for each metric
def normalize_metrics(df, metrics):
    df_norm = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if metric in ['DRD', 'NRM', 'MPM']:
            df_norm[f'{metric}_norm'] = 1 - (df[metric] - min_val) / (max_val - min_val)
        else:
            df_norm[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
    return df_norm

# dpi=300 for high quality
# bbox_inches='tight' ensures that the plot is saved without extra whitespace
def save_plot(path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def create_quality_analysis(df, category, category_dir):
    cat_df = df[df['Dataset'] == category]
    
    metrics = []
    for m in ['PSNR', 'F-measure', 'PF-measure', 'DRD', 'NRM', 'MPM']:
        if m in cat_df.columns:
            metrics.append(m)
    
    # print(metrics)
    df_norm = normalize_metrics(cat_df, metrics)
    
    norm_columns = []
    for m in metrics:
        norm_columns.append(f'{m}_norm')
    df_norm['Quality_Score'] = df_norm[norm_columns].mean(axis=1)
    
    models = cat_df['Model'].tolist()

    quality_scores = []
    runtimes = []
    
    # collect normalized quality scores and runtimes 
    for model_name in models:
        model_row_normalized = df_norm[df_norm['Model'] == model_name]
        model_row_original = cat_df[cat_df['Model'] == model_name]

        quality_score = model_row_normalized['Quality_Score'].iloc[0]
        runtime_seconds = model_row_original['Model Runtime (s)'].iloc[0]

        quality_scores.append(quality_score)
        runtimes.append(runtime_seconds)
    
    # runtime normalization
    min_rt, max_rt = min(runtimes), max(runtimes)
    speed_scores = []
    for rt in runtimes:
        speed_score = (max_rt - rt) / (max_rt - min_rt)
        speed_scores.append(speed_score)
    
    # sort and group results by quality and speed scores
    data = sorted(zip(models, quality_scores, speed_scores, runtimes))
    s_models, s_quality, s_speed, s_runtime = zip(*data)
    
    # fig = the whole figure, ax1 and ax2 are the two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # create lines for quality and speed and compare them
    for i, (model, quality, speed, runtime) in enumerate(data):
        ax1.vlines(i - 0.15, 0, quality, colors=COLORS['quality'], linewidth=6, alpha=0.8)
        ax1.text(i - 0.15, quality + 0.03, f'{quality:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=18, color=COLORS['quality'])
        ax1.vlines(i + 0.15, 0, speed, colors=COLORS['speed'], linewidth=6, alpha=0.8)
        ax1.text(i + 0.15, speed + 0.03, f'{runtime:.1f}s', ha='center', va='bottom',
                fontweight='bold', fontsize=18, color=COLORS['speed'])
    
    # create the bar chart for quality scores
    ax1.set_xticks(range(len(s_models)))
    ax1.set_xticklabels(s_models, rotation=0, ha='center', fontweight='bold', fontsize=16)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlim(-0.5, len(s_models) - 0.5)
    ax1.grid(True, alpha=0.3)
    
    # adjust fonts
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(16) 
    for label in ax1.get_xticklabels():
        label.set_color(COLORS['text'])
        label.set_fontweight('bold')
    
    # calculate efficiency scores as a ratio of quality and runtime
    efficiency_scores = []
    for q, r in zip(quality_scores, runtimes):
        efficiency_scores.append(q/r)
    
    def sort_by_efficiency_score(item):
        return item[1]
    
    eff_data = sorted(zip(models, efficiency_scores), key=sort_by_efficiency_score, reverse=True)
    eff_models, eff_scores = zip(*eff_data)
    
    eff_colors = []
    for m in eff_models:
        eff_colors.append(COLORS[m])
    
    # create the bar chart for efficiency scores
    ax2.bar(eff_models, eff_scores, color=eff_colors, alpha=0.8, edgecolor=COLORS['text'], linewidth=1)
    ax2.set_ylim(0, max(eff_scores) * 1.15)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=0)
    
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(16)
    
    for label in ax2.get_xticklabels():
        label.set_color(COLORS['text'])
        label.set_fontweight('bold')
        label.set_fontsize(16)
    
    plt.suptitle(f'Analiza asupra eficienței și performanței pentru {category.replace("_", " ").title()}', 
                fontsize=18, fontweight='bold')
    save_plot(category_dir / f"{category.replace('_', ' ').title()}_quality_analysis.png", 400)

def create_correlation_heatmap(df, plots_dir, timestamp):
    metrics = ['PSNR', 'F-measure', 'PF-measure', 'DRD', 'NRM', 'MPM']
    
    corr_data = df[metrics].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.grid(False)
    
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{corr_data.iloc[i, j]:.3f}', ha='center', va='center',
                   fontweight='bold', fontsize=14, color=COLORS['text_inv'])
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Coeficient de corelație', rotation=270, labelpad=20, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color(COLORS['border'])
    
    save_plot(plots_dir / f"global_correlation_heatmap_{timestamp}.png")

def create_efficiency_table(df, category, category_dir):
    cat_df = df[df['Dataset'] == category]
    
    metrics = []
    for m in ['PSNR', 'F-measure', 'PF-measure', 'DRD', 'NRM', 'MPM']:
        if m in cat_df.columns:
            metrics.append(m)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    header = ['Model'] + metrics + ['Runtime (s)']
    
    best_vals = {}
    for m in metrics:
        if m in ['DRD', 'NRM', 'MPM']:
            best_vals[m] = cat_df[m].min()
        else:
            best_vals[m] = cat_df[m].max()
    
    table_data = []
    for model in cat_df['Model']:
        row = [model]
        for metric in metrics:
            val = cat_df[cat_df['Model'] == model][metric].iloc[0]
            if metric in ['DRD', 'NRM', 'MPM']:
                row.append(f'{val:.6f}')
            else:
                row.append(f'{val:.3f}')
        runtime = cat_df[cat_df['Model'] == model]['Model Runtime (s)'].iloc[0]
        row.append(f'{runtime:.1f}')
        table_data.append(row)
    
    header_colors = []
    for _ in range(len(header)):
        header_colors.append(COLORS['header'])
    
    table = ax.table(cellText=table_data, colLabels=header, cellLoc='center', loc='center',
                    colColours=header_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.5)
    
    # Style table
    for j in range(len(header)):
        table[(0, j)].set_text_props(weight='bold', color=COLORS['text_inv'])
        table[(0, j)].set_height(0.12)
        table[(0, j)].set_width(0.12)
    
    for i, model in enumerate(cat_df['Model']):
        table[(i+1, 0)].set_facecolor(COLORS['cell'])
        table[(i+1, 0)].set_text_props(weight='bold', color=COLORS['text'], size=14)
        table[(i+1, 0)].set_height(0.10)
        table[(i+1, 0)].set_width(0.12)
        
        for j, metric in enumerate(metrics):
            val = cat_df[cat_df['Model'] == model][metric].iloc[0]
            cell = table[(i+1, j+1)]
            cell.set_height(0.10)
            cell.set_width(0.12)
            if abs(val - best_vals[metric]) < 1e-10:
                cell.set_text_props(weight='bold', size=14)
                cell.set_facecolor(COLORS['best'])
            else:
                cell.set_facecolor(COLORS['cell'])
                cell.set_text_props(size=14)
        
        table[(i+1, len(metrics)+1)].set_facecolor(COLORS['runtime'])
        table[(i+1, len(metrics)+1)].set_text_props(weight='bold', color=COLORS['text_inv'], size=14)
        table[(i+1, len(metrics)+1)].set_height(0.10)
        table[(i+1, len(metrics)+1)].set_width(0.12)
    
    save_plot(category_dir / f"{category.replace('_', ' ').title()}_efficiency_analysis.png")

def create_model_insights(df, category, category_dir):
    cat_df = df[df['Dataset'] == category]
    
    metrics = []
    for m in ['PSNR', 'F-measure', 'PF-measure', 'DRD', 'NRM', 'MPM']:
        if m in cat_df.columns:
            metrics.append(m)
    
    df_norm = normalize_metrics(cat_df, metrics)
    
    norm_columns = []
    for m in metrics:
        norm_columns.append(f'{m}_norm')
    df_norm['Quality_Score'] = df_norm[norm_columns].mean(axis=1)
    sorted_df = df_norm.sort_values('Quality_Score', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    model_colors = []
    for m in sorted_df['Model']:
        model_colors.append(COLORS[m])
    
    bars = ax.bar(sorted_df['Model'], sorted_df['Quality_Score'],
                 color=model_colors, alpha=0.8, edgecolor=COLORS['text'], linewidth=1)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'#{i+1}\n{height:.3f}', ha='center', va='bottom', 
               fontweight='bold', fontsize=14)
    
    ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right', fontweight='bold')
    ax.set_ylabel('Scor de calitate', fontweight='bold', fontsize=12)
    ax.set_ylim(0, max(sorted_df['Quality_Score']) * 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Make y-axis labels bold
    ax.tick_params(axis='y', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    save_plot(category_dir / f"{category.replace('_', ' ').title()}_model_insights.png")

def create_global_charts(df, plots_dir, timestamp):
    metrics = []
    for m in ['PSNR', 'F-measure', 'PF-measure', 'DRD', 'NRM', 'MPM']:
        if m in df.columns:
            metrics.append(m)
    
    models = sorted(df['Model'].unique())
    categories = sorted(df['Dataset'].unique())
    
    model_scores = {}
    for model in models:
        model_scores[model] = []
    
    for model in models:
        for category in categories:
            cat_model_df = df[(df['Model'] == model) & (df['Dataset'] == category)]
            if len(cat_model_df) > 0:
                cat_df = df[df['Dataset'] == category]
                scores = []
                for metric in metrics:
                    val = cat_model_df[metric].iloc[0]
                    min_val, max_val = cat_df[metric].min(), cat_df[metric].max()
                    if max_val != min_val:
                        if metric in ['DRD', 'NRM', 'MPM']:
                            norm = 1 - (val - min_val) / (max_val - min_val)
                        else:
                            norm = (val - min_val) / (max_val - min_val)
                    else:
                        norm = 0.5
                    scores.append(norm)
                
                quality = np.mean(scores)
                runtime = cat_model_df['Model Runtime (s)'].iloc[0]
                model_scores[model].extend([quality, quality/runtime])
    
    # Quality and Efficiency charts
    for chart_type, idx in [('quality', 0), ('efficiency', 1)]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        avg_scores = {}
        for m in models:
            relevant_scores = []
            for i in range(idx, len(model_scores[m]), 2):
                relevant_scores.append(model_scores[m][i])
            avg_scores[m] = np.mean(relevant_scores)
        
        def sort_by_avg_score(model):
            return avg_scores[model]
        
        sorted_models = sorted(models, key=sort_by_avg_score, reverse=True)
        
        values = []
        for m in sorted_models:
            values.append(avg_scores[m])
        
        chart_colors = []
        for m in sorted_models:
            chart_colors.append(COLORS[m])
        
        bars = ax.bar(sorted_models, values, color=chart_colors, 
                     alpha=0.8, edgecolor=COLORS['text'], linewidth=1)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'#{i+1}\n{height:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=14)
        
        if chart_type == "quality":
            ylabel = 'Scor de calitate'
        else:
            ylabel = 'Scor de eficiență'
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Make labels bold
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for label in ax.get_xticklabels():
            label.set_color(COLORS['text'])
            label.set_fontweight('bold')
        
        save_plot(plots_dir / f"global_{chart_type}_ranking_{timestamp}.png")

def main():
    df = pd.read_csv(BASE_DIR / "outputs" / "metrics" / "results_og.csv")
    if not all(col in df.columns for col in ['Model', 'Dataset', 'PSNR', 'F-measure']):
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    categories = sorted(df['Dataset'].unique())
    
    create_correlation_heatmap(df, PLOTS_DIR, timestamp)
    create_global_charts(df, PLOTS_DIR, timestamp)
    
    for category in categories:
        cat_dir = PLOTS_DIR / f"category_{category.replace(' ', '_')}_{timestamp}"
        cat_dir.mkdir(exist_ok=True)
        
        create_quality_analysis(df, category, cat_dir)
        create_efficiency_table(df, category, cat_dir)
        create_model_insights(df, category, cat_dir)

if __name__ == "__main__":
    main()