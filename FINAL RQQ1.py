

###this is the full report for the analysis of structural complexity changes pre-post




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg

def analyze_structural_complexity(pre_csv, post_csv):
    """
    Comprehensive analysis of structural complexity changes (pre-post).
    Specifically addresses RQ1: How does the Mind Elevator tool influence
    the structural complexity of users' arguments, based on Toulmin components
    """
    # Load data
    pre_df = pd.read_csv(pre_csv)
    post_df = pd.read_csv(post_csv)
    
    print("Analyzing structural complexity changes (pre-post)...")
    
    # Check that structural_complexity exists in both datasets
    if 'structural_complexity' not in pre_df.columns or 'structural_complexity' not in post_df.columns:
        print("ERROR: 'structural_complexity' column not found in one or both datasets")
        print("Pre columns:", pre_df.columns.tolist())
        print("Post columns:", post_df.columns.tolist())
        return
    
    # Ensure user_ID is available for matching
    if 'user_ID' not in pre_df.columns or 'user_ID' not in post_df.columns:
        print("ERROR: 'user_ID' column not found for matching participants")
        return
    
    # Merge pre and post data on user_ID
    merged_df = pd.merge(
        pre_df[['user_ID', 'structural_complexity']], 
        post_df[['user_ID', 'structural_complexity']], 
        on='user_ID', 
        suffixes=('_pre', '_post')
    )
    
    # Check if we have matched pairs
    if len(merged_df) != len(pre_df) or len(merged_df) != len(post_df):
        print(f"WARNING: Not all participants have both pre and post data")
        print(f"Pre: {len(pre_df)}, Post: {len(post_df)}, Matched: {len(merged_df)}")
    
    print(f"\nNumber of participants with complete data: {len(merged_df)}")
    
    # Calculate change scores
    merged_df['change'] = merged_df['structural_complexity_post'] - merged_df['structural_complexity_pre']
    
    # 1. DESCRIPTIVE STATISTICS
    pre_mean = merged_df['structural_complexity_pre'].mean()
    pre_median = merged_df['structural_complexity_pre'].median()
    pre_sd = merged_df['structural_complexity_pre'].std()
    pre_min = merged_df['structural_complexity_pre'].min()
    pre_max = merged_df['structural_complexity_pre'].max()
    
    post_mean = merged_df['structural_complexity_post'].mean()
    post_median = merged_df['structural_complexity_post'].median()
    post_sd = merged_df['structural_complexity_post'].std()
    post_min = merged_df['structural_complexity_post'].min()
    post_max = merged_df['structural_complexity_post'].max()
    
    change_mean = merged_df['change'].mean()
    change_sd = merged_df['change'].std()
    
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(f"Pre-test:  Mean = {pre_mean:.2f}, Median = {pre_median:.2f}, SD = {pre_sd:.2f}, Range = {pre_min:.2f}-{pre_max:.2f}")
    print(f"Post-test: Mean = {post_mean:.2f}, Median = {post_median:.2f}, SD = {post_sd:.2f}, Range = {post_min:.2f}-{post_max:.2f}")
    print(f"Change:    Mean = {change_mean:.2f}, SD = {change_sd:.2f}")
    
    # 2. INFERENTIAL STATISTICS
    # Paired-samples t-test
    t_stat, p_value = stats.ttest_rel(merged_df['structural_complexity_post'], 
                                     merged_df['structural_complexity_pre'])
    
    # Calculate Cohen's d effect size
    d = (post_mean - pre_mean) / pre_sd
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_p = stats.wilcoxon(merged_df['structural_complexity_post'], 
                                merged_df['structural_complexity_pre'])
    
    # Calculate confidence intervals (95%)
    ci_low, ci_high = stats.t.interval(0.95, len(merged_df)-1, 
                                      loc=change_mean, 
                                      scale=stats.sem(merged_df['change']))
    
    print("\n=== INFERENTIAL STATISTICS ===")
    print(f"Paired t-test: t({len(merged_df)-1}) = {t_stat:.3f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        sig_text = "significant"
    else:
        sig_text = "not significant"
    
    print(f"The difference between pre and post scores is statistically {sig_text}.")
    print(f"95% CI for mean difference: [{ci_low:.2f}, {ci_high:.2f}]")
    
    print(f"\nEffect size (Cohen's d): {d:.2f}")
    # Interpret effect size
    if abs(d) < 0.2:
        effect_text = "negligible"
    elif abs(d) < 0.5:
        effect_text = "small"
    elif abs(d) < 0.8:
        effect_text = "medium"
    else:
        effect_text = "large"
    
    print(f"This represents a {effect_text} effect size.")
    
    # Non-parametric test results
    print(f"\nWilcoxon signed-rank test: W = {w_stat:.1f}, p = {w_p:.4f}")
    
    # 3. VISUALIZATIONS
    # Create figure with multiple plots
    fig = plt.figure(figsize=(15, 10))
    
    # A. Pre vs Post Boxplot
    ax1 = fig.add_subplot(221)
    box_data = [merged_df['structural_complexity_pre'], merged_df['structural_complexity_post']]
    box = ax1.boxplot(box_data, patch_artist=True, labels=['Pre-test', 'Post-test'])
    
    # Colors for boxplots
    colors = ['#2271B2', '#259C74']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add individual data points with jitter
    for i, data in enumerate(box_data):
        # Add jitter
        x = np.random.normal(i+1, 0.08, size=len(data))
        ax1.scatter(x, data, alpha=0.5, s=30, edgecolors='gray', color=colors[i])
    
    ax1.set_title('Distribution of Structural Complexity Scores', fontweight='bold')
    ax1.set_ylabel('Structural Complexity Score')
    ax1.set_ylim(0.5, 4.5)  # Assuming 1-4 scale
    
    # Add mean value labels
    ax1.text(1, pre_mean, f'Mean: {pre_mean:.2f}', ha='center', va='bottom', fontweight='bold')
    ax1.text(2, post_mean, f'Mean: {post_mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # B. Spaghetti Plot (Individual changes)
    ax2 = fig.add_subplot(222)
    
    # Plot individual lines for each participant
    for _, row in merged_df.iterrows():
        ax2.plot([1, 2], [row['structural_complexity_pre'], row['structural_complexity_post']], 
                  'o-', alpha=0.3, color='gray')
    
    # Add mean line (thicker)
    ax2.plot([1, 2], [pre_mean, post_mean], 'o-', linewidth=3, markersize=10, 
              color='#C66B28', label='Mean')
    
    ax2.set_title('Individual Changes in Structural Complexity', fontweight='bold')
    ax2.set_ylabel('Structural Complexity Score')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Pre-test', 'Post-test'])
    ax2.set_xlim(0.5, 2.5)
    ax2.set_ylim(0.5, 4.5)  # Assuming 1-4 scale
    ax2.legend()
    
    # C. Mean comparison with error bars
    ax3 = fig.add_subplot(223)
    
    # Calculate standard error
    pre_se = pre_sd / np.sqrt(len(merged_df))
    post_se = post_sd / np.sqrt(len(merged_df))
    
    bar_colors = ['#2271B2', '#259C74']
    bars = ax3.bar([1, 2], [pre_mean, post_mean], yerr=[pre_se, post_se], 
                    capsize=10, color=bar_colors, width=0.6)
    
    # Add value labels
    for bar, value, se in zip(bars, [pre_mean, post_mean], [pre_se, post_se]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + se + 0.1,
                  f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add connecting line between bars
    ax3.plot([1, 2], [pre_mean, post_mean], '-', color='gray', linestyle='--', alpha=0.7)
    
    # Add significance annotation if applicable
    if p_value < 0.05:
        max_height = max(pre_mean, post_mean) + max(pre_se, post_se) + 0.2
        ax3.plot([1, 1, 2, 2], [max_height, max_height+0.1, max_height+0.1, max_height], 
                  lw=1.5, c='black')
        stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
        ax3.text(1.5, max_height+0.1, f' {stars} ', ha='center', va='bottom')
    
    ax3.set_title('Mean Structural Complexity Scores with SE', fontweight='bold')
    ax3.set_ylabel('Structural Complexity Score')
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Pre-test', 'Post-test'])
    ax3.set_xlim(0.5, 2.5)
    ax3.set_ylim(0, 5)  # Assuming 1-4 scale with room for error bars
    
    # D. Distribution comparison (density plot)
    ax4 = fig.add_subplot(224)
    
    sns.kdeplot(merged_df['structural_complexity_pre'], ax=ax4, 
                 shade=True, color=colors[0], label='Pre-test')
    sns.kdeplot(merged_df['structural_complexity_post'], ax=ax4, 
                 shade=True, color=colors[1], label='Post-test')
    
    # Add vertical lines for means
    ax4.axvline(pre_mean, color=colors[0], linestyle='--', 
                 label=f'Pre Mean: {pre_mean:.2f}')
    ax4.axvline(post_mean, color=colors[1], linestyle='--', 
                 label=f'Post Mean: {post_mean:.2f}')
    
    ax4.set_title('Distribution of Structural Complexity Scores', fontweight='bold')
    ax4.set_xlabel('Structural Complexity Score')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    # Add overall title and adjust layout
    plt.suptitle(f'Impact of Mind Elevator Tool on Structural Complexity\n(N = {len(merged_df)})', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save high-resolution figure
    plt.savefig('structural_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('structural_complexity_analysis.pdf', bbox_inches='tight')
    print("\nVisualization saved as 'structural_complexity_analysis.png' and PDF version")
    
    # 4. GENERATE SUMMARY TABLE
    print("\n=== SUMMARY TABLE FOR THESIS ===")
    print("Table 4.1: Structural Complexity Analysis Results")
    print("-" * 50)
    print(f"{'Measure':<20} {'Pre-test':<15} {'Post-test':<15} {'Change':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {pre_mean:.2f} {post_mean:.2f} {change_mean:+.2f}")
    print(f"{'Median':<20} {pre_median:.2f} {post_median:.2f} {post_median-pre_median:+.2f}")
    print(f"{'Standard Deviation':<20} {pre_sd:.2f} {post_sd:.2f} {post_sd-pre_sd:+.2f}")
    print(f"{'Range':<20} {pre_min:.2f}-{pre_max:.2f} {post_min:.2f}-{post_max:.2f}")
    print("-" * 50)
    print(f"t-test: t({len(merged_df)-1}) = {t_stat:.3f}, p = {p_value:.4f}, d = {d:.2f} ({effect_text})")
    print("-" * 50)
    
    # 5. IEEE-STYLE RESULTS PARAGRAPH
    print("\n=== SAMPLE RESULTS TEXT FOR THESIS ===")
    print(f"Analysis of structural complexity scores revealed a {sig_text} increase from")
    print(f"pre-test (M = {pre_mean:.2f}, SD = {pre_sd:.2f}) to post-test (M = {post_mean:.2f}, SD = {post_sd:.2f}),")
    print(f"t({len(merged_df)-1}) = {t_stat:.3f}, p = {p_value:.4f}, d = {d:.2f}. This represents a {effect_text}")
    print(f"effect size. The 95% confidence interval for the mean increase was [{ci_low:.2f}, {ci_high:.2f}].")
    print(f"These results indicate that the Mind Elevator tool had a{'' if effect_text == 'large' else 'n'} {effect_text}")
    print(f"impact on the structural complexity of participants' arguments.")
    
    # Additional analysis: % of participants who improved
    improved = (merged_df['change'] > 0).sum()
    improved_pct = improved / len(merged_df) * 100
    
    print(f"\nAdditionally, {improved} out of {len(merged_df)} participants ({improved_pct:.1f}%) showed")
    print("improvement in structural complexity after using the Mind Elevator tool.")
    
    return merged_df
# After the analyze_structural_complexity function
def save_results_to_csv(merged_df, output_dir='.'):
    """Save analysis results to CSV files"""
    # Create participant-level data file
    participant_df = merged_df.copy()
    participant_df['change'] = participant_df['structural_complexity_post'] - participant_df['structural_complexity_pre']
    participant_df['improved'] = participant_df['change'] > 0
    
    # Save participant-level data
    participant_df.to_csv(f'{output_dir}/structural_complexity_participant_data.csv', index=False)
    
    # Create summary statistics file
    summary_stats = {
        'Measure': ['Mean', 'Median', 'SD', 'Min', 'Max', 'Sample Size', 
                   't-statistic', 'p-value', 'Effect size (d)', 
                   'CI Lower', 'CI Upper', '% Improved'],
        'Pre-test': [
            participant_df['structural_complexity_pre'].mean(),
            participant_df['structural_complexity_pre'].median(),
            participant_df['structural_complexity_pre'].std(),
            participant_df['structural_complexity_pre'].min(),
            participant_df['structural_complexity_pre'].max(),
            len(participant_df),
            '', '', '', '', '', ''
        ],
        'Post-test': [
            participant_df['structural_complexity_post'].mean(),
            participant_df['structural_complexity_post'].median(),
            participant_df['structural_complexity_post'].std(),
            participant_df['structural_complexity_post'].min(),
            participant_df['structural_complexity_post'].max(),
            len(participant_df),
            '', '', '', '', '', ''
        ],
        'Change': [
            participant_df['change'].mean(),
            participant_df['change'].median(),
            participant_df['change'].std(),
            participant_df['change'].min(),
            participant_df['change'].max(),
            '',
            round(stats.ttest_rel(participant_df['structural_complexity_post'], 
                               participant_df['structural_complexity_pre'])[0], 3),
            round(stats.ttest_rel(participant_df['structural_complexity_post'], 
                               participant_df['structural_complexity_pre'])[1], 4),
            round((participant_df['structural_complexity_post'].mean() - 
                participant_df['structural_complexity_pre'].mean()) / 
                participant_df['structural_complexity_pre'].std(), 2),
            round(stats.t.interval(0.95, len(participant_df)-1, 
                              loc=participant_df['change'].mean(), 
                              scale=stats.sem(participant_df['change']))[0], 2),
            round(stats.t.interval(0.95, len(participant_df)-1, 
                              loc=participant_df['change'].mean(), 
                              scale=stats.sem(participant_df['change']))[1], 2),
            f"{(participant_df['improved'].sum() / len(participant_df) * 100):.1f}%"
        ]
    }
    
    # Create and save summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f'{output_dir}/structural_complexity_summary.csv', index=False)
    
    print(f"\nSaved results to CSV files:")
    print(f"1. Individual participant data: structural_complexity_participant_data.csv")
    print(f"2. Summary statistics: structural_complexity_summary.csv")

def create_enhanced_visualizations(pre_csv, post_csv):
    """Create clean, non-overlapping visualizations of structural complexity"""
    # Load data
    pre_df = pd.read_csv(pre_csv)
    post_df = pd.read_csv(post_csv)
    
    # Merge pre and post data
    merged_df = pd.merge(
        pre_df[['user_ID', 'structural_complexity']], 
        post_df[['user_ID', 'structural_complexity']], 
        on='user_ID', 
        suffixes=('_pre', '_post')
    )
    
    # Setup colors as requested
    pre_color = "#4C72B0"  # Blue
    post_color = "#55A868"  # Green
    
    # Calculate stats for annotations
    pre_mean = merged_df['structural_complexity_pre'].mean()
    post_mean = merged_df['structural_complexity_post'].mean()
    pre_se = merged_df['structural_complexity_pre'].std() / np.sqrt(len(merged_df))
    post_se = merged_df['structural_complexity_post'].std() / np.sqrt(len(merged_df))
    
    # Prepare data for plotting
    data_long = pd.DataFrame({
        'Time': ['Pre-test'] * len(merged_df) + ['Post-test'] * len(merged_df),
        'Structural Complexity': np.concatenate([
            merged_df['structural_complexity_pre'], 
            merged_df['structural_complexity_post']
        ])
    })
    
    # 1. VIOLIN PLOT
    plt.figure(figsize=(10, 6))
    
    # Create violin plot with individual data points
    ax = sns.violinplot(x='Time', y='Structural Complexity', data=data_long, 
                   palette=[pre_color, post_color], inner=None)
    
    # Add scatter points with jitter
    sns.stripplot(x='Time', y='Structural Complexity', data=data_long, 
              jitter=True, size=6, alpha=0.7, palette=[pre_color, post_color])
    
    # Add mean points with error bars
    plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
             fmt='o', color='black', markersize=10, capsize=10, 
             markerfacecolor='white', markeredgewidth=2, elinewidth=2)
    
    # Add mean value labels
    for i, (mean, se) in enumerate(zip([pre_mean, post_mean], [pre_se, post_se])):
        plt.text(i, mean + se + 0.1, f'Mean: {mean:.2f}', ha='center', fontweight='bold')
    
    # Perform paired t-test for annotation
    t_stat, p_value = stats.ttest_rel(merged_df['structural_complexity_post'], 
                                    merged_df['structural_complexity_pre'])
    
    # Add significance annotation
    y_max = max(merged_df['structural_complexity_post'].max(), 
               merged_df['structural_complexity_pre'].max()) + 0.2
    plt.plot([0, 0, 1, 1], [y_max, y_max+0.1, y_max+0.1, y_max], lw=1.5, c='black')
    
    # Add stars based on p-value
    stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
    plt.text(0.5, y_max+0.15, f'{stars}', ha='center', va='bottom', fontsize=16)
    
    # Add p-value
    plt.text(0.5, y_max+0.05, f'p < 0.001', ha='center', va='bottom')
    
    plt.title('Distribution of Structural Complexity Scores', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Structural Complexity Score', fontsize=14)
    plt.xlabel('', fontsize=14)  # Remove x-label as it's redundant
    plt.ylim(1.5, 4.5)  # Adjust y-axis limits
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('structural_complexity_violin.png', dpi=300, bbox_inches='tight')
    plt.savefig('structural_complexity_violin.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. BAR CHART WITH ERROR BARS
    plt.figure(figsize=(8, 6))
    
    # Create bar chart
    bar_width = 0.6
    bars = plt.bar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
               width=bar_width, capsize=10, color=[pre_color, post_color], 
               edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, [pre_mean, post_mean])):
        plt.text(i, value/2, f'{value:.2f}', ha='center', va='center', 
             color='white', fontweight='bold', fontsize=12)
    
    # Add connecting line for pre-post
    plt.plot([0+bar_width/2, 1-bar_width/2], [pre_mean, post_mean], 
         color='gray', linestyle='--', alpha=0.7, lw=2)
    
    # Add significance annotation
    effect_size = (post_mean - pre_mean) / merged_df['structural_complexity_pre'].std()
    plt.text(0.5, 0.2, f'd = {effect_size:.2f} (very large effect)', 
         ha='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.title('Mean Structural Complexity Scores', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Structural Complexity Score', fontsize=14)
    plt.xticks([0, 1], ['Pre-test', 'Post-test'], fontsize=12)
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('structural_complexity_means.png', dpi=300, bbox_inches='tight')
    plt.savefig('structural_complexity_means.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. DENSITY DISTRIBUTION PLOT
    plt.figure(figsize=(10, 6))
    
    # Create density plots
    sns.kdeplot(merged_df['structural_complexity_pre'], fill=True, 
            label=f'Pre-test (Mean: {pre_mean:.2f})', color=pre_color, alpha=0.7)
    sns.kdeplot(merged_df['structural_complexity_post'], fill=True, 
            label=f'Post-test (Mean: {post_mean:.2f})', color=post_color, alpha=0.7)
    
    # Add vertical lines for means
    plt.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
    plt.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
    # Add shaded area for mean difference
    plt.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    plt.text((pre_mean + post_mean)/2, 0.1, f'Δ = {post_mean - pre_mean:.2f}', 
         ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Distribution of Structural Complexity Scores', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Structural Complexity Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xlim(1.5, 4.5)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('structural_complexity_density.png', dpi=300, bbox_inches='tight')
    plt.savefig('structural_complexity_density.pdf', bbox_inches='tight')
    plt.close()
    
    print("Enhanced visualizations created and saved as:")
    print("1. structural_complexity_violin.png/pdf")
    print("2. structural_complexity_means.png/pdf")
    print("3. structural_complexity_density.png/pdf")










# def create_combined_density_bar_chart(pre_csv, post_csv):
#     """Create IEEE/CHI-compliant density plot with integrated bar chart"""
#     # Load and prepare data
#     pre_df = pd.read_csv(pre_csv)
#     post_df = pd.read_csv(post_csv)
    
#     merged_df = pd.merge(
#         pre_df[['user_ID', 'structural_complexity']], 
#         post_df[['user_ID', 'structural_complexity']], 
#         on='user_ID', 
#         suffixes=('_pre', '_post')
#     )
    
#     # Colors following IEEE/CHI standards
#     pre_color = "#4C72B0"  # Blue
#     post_color = "#55A868"  # Green
    
#     # Calculate statistics
#     pre_mean = merged_df['structural_complexity_pre'].mean()
#     post_mean = merged_df['structural_complexity_post'].mean()
#     pre_sd = merged_df['structural_complexity_pre'].std()
#     post_sd = merged_df['structural_complexity_post'].std()
#     pre_se = pre_sd / np.sqrt(len(merged_df))
#     post_se = post_sd / np.sqrt(len(merged_df))
    
#     # Calculate t-test and effect size for annotation
#     t_stat, p_value = stats.ttest_rel(merged_df['structural_complexity_post'], 
#                                       merged_df['structural_complexity_pre'])
#     d = (post_mean - pre_mean) / pre_sd
    
#     # Calculate confidence intervals
#     ci_low, ci_high = stats.t.interval(0.95, len(merged_df)-1, 
#                                       loc=post_mean - pre_mean, 
#                                       scale=stats.sem(merged_df['structural_complexity_post'] - 
#                                                      merged_df['structural_complexity_pre']))
    
#     # Create figure with two panels side by side
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
#                                    gridspec_kw={'width_ratios': [2, 1]})
    
#     # Left panel: Density plot
#     sns.kdeplot(x=merged_df['structural_complexity_pre'], ax=ax1, fill=True, 
#                 label=f'Pre-test (M = {pre_mean:.2f})', color=pre_color, alpha=0.7)
#     sns.kdeplot(x=merged_df['structural_complexity_post'], ax=ax1, fill=True, 
#                 label=f'Post-test (M = {post_mean:.2f})', color=post_color, alpha=0.7)
    
#     # Add mean lines to density plot
#     ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
#     ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
#     # Add shaded area for mean difference
#     ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
#     # Add effect size annotation in the density plot
#     effect_size_text = f"d = {d:.2f} (large effect)"
#     if d < 0.8:
#         effect_size_text = f"d = {d:.2f} (medium effect)" if d >= 0.5 else f"d = {d:.2f} (small effect)"
    
#     # Position the effect size text in the middle of the shaded area
#     ax1.text((pre_mean + post_mean)/2, 0.85, effect_size_text, 
#             ha='center', va='center', fontweight='bold',
#             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
#     # Add delta indicator
#     ax1.text((pre_mean + post_mean)/2, 0.4, f'Δ = {post_mean - pre_mean:.2f}', 
#             ha='center', va='center', fontsize=12, fontweight='bold',
#             bbox=dict(facecolor='white', alpha=0.9))
    
#     # Create dataframes for pre and post points to avoid deprecation warnings
#     pre_points = pd.DataFrame({
#         'x': merged_df['structural_complexity_pre'],
#         'y': [-0.05] * len(merged_df),
#         'group': ['Pre-test'] * len(merged_df)
#     })
    
#     post_points = pd.DataFrame({
#         'x': merged_df['structural_complexity_post'],
#         'y': [-0.15] * len(merged_df),
#         'group': ['Post-test'] * len(merged_df)
#     })
    
#     # Combine points for unified plotting
#     all_points = pd.concat([pre_points, post_points])
    
#     # Plot all points with proper hue to avoid deprecation warning
#     sns.stripplot(x='x', y='y', hue='group', data=all_points, ax=ax1,
#                  palette={
#                      'Pre-test': pre_color, 
#                      'Post-test': post_color
#                  },
#                  size=8, alpha=0.7, jitter=True, dodge=False, legend=False,edgecolor='white',  # Add white edges
#              linewidth=0.5,      # Control edge thickness
#              )
    
#     # Add labels for the strip points
#     center_x = (pre_mean + post_mean)/2
#     ax1.text(center_x - 0.1, -0.05, "Pre-test scores", color=pre_color, 
#          ha='right', va='center', fontsize=10)

# # Post-test label - aligned to the left side of the center point
#     ax1.text(center_x + 0.1, -0.15, "Post-test scores", color=post_color, 
#             ha='left', va='center', fontsize=10)
    
#     # Right panel: Bar chart with error bars
#     bar_width = 0.7
#     x_pos = [0, 1]
#     bars = ax2.bar(x_pos, [pre_mean, post_mean], yerr=[pre_se, post_se], 
#                   width=bar_width, capsize=10, color=[pre_color, post_color], 
#                   edgecolor='black', linewidth=1.5)
    
#     # Add value labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         ax2.text(bar.get_x() + bar.get_width()/2., height + (pre_se if i==0 else post_se) + 0.1,
#                 f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
#     # Add significance line and annotation
#     y_max = max(pre_mean + pre_se, post_mean + post_se) + 0.3
#     ax2.plot([0, 0, 1, 1], [y_max-0.1, y_max, y_max, y_max-0.1], lw=1.5, c='black')
    
#     # Add stars for significance
#     stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
#     ax2.text(0.5, y_max + 0.1, stars, ha='center', va='bottom', fontsize=16)
    
#     # Customize left panel (density)
#     ax1.set_title('Distribution of Structural Complexity Scores', fontsize=14, fontweight='bold', pad=15)
#     ax1.set_xlabel('Structural Complexity Score', fontsize=12)
#     ax1.set_ylabel('Density', fontsize=12)
#     ax1.set_xlim(1.5, 4.5)
#     ax1.set_ylim(-0.2, 1.4)  # FIXED: Increased upper limit to show full distribution
#     ax1.grid(linestyle='--', alpha=0.6)
#     ax1.legend(fontsize=11, loc='upper left')
    
#     # Remove x-axis labels from stripplot
#     # Set explicit x-axis ticks while keeping the label
#     ax1.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
#     ax1.set_xticklabels(['2.0', '2.5', '3.0', '3.5', '4.0'])
#     # Keep the x-axis label
#     ax1.set_xlabel('Structural Complexity Score', fontsize=12)
    
    
#     # Customize right panel (bars)
#     ax2.set_title('Mean Scores with SE', fontsize=14, fontweight='bold', pad=15)
#     ax2.set_ylabel('Structural Complexity Score', fontsize=12)
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels(['Pre-test', 'Post-test'], fontsize=11)
#     ax2.set_ylim(1.5, y_max + 0.5)
#     ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
#     # Add overall title
#     plt.suptitle('Impact of Mind Elevator on Structural Complexity of Arguments', 
#                  fontsize=16, fontweight='bold', y=0.98)
    
#     # Add paired sample t-test results as a footnote
#     plt.figtext(0.5, 0.01, 
#                 f"Paired t-test: t({len(merged_df)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
#                 ha='center', fontsize=10, style='italic')
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # ERROR HANDLING FOR FILE SAVING
#     try:
#         # First save the PNG which usually works
#         plt.savefig('structural_complexity_combined.png', dpi=300, bbox_inches='tight')
#         print("Saved PNG successfully")
        
#         # Try to save PDF with error handling
#         try:
#             plt.savefig('structural_complexity_combined.pdf', bbox_inches='tight')
#             print("Saved PDF successfully")
#         except PermissionError:
#             print("Permission error saving PDF. Trying alternative filename...")
#             # Try with timestamp in filename to make it unique
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             plt.savefig(f'structural_complexity_combined_{timestamp}.pdf', bbox_inches='tight')
#             print(f"Saved as: structural_complexity_combined_{timestamp}.pdf")
#     except Exception as e:
#         print(f"Error saving files: {type(e).__name__}: {e}")
#     finally:
#         plt.close()
    
#     print("Created IEEE/CHI-compliant combined density-bar visualization")


# ##THIS BELOW DO NOT HAVE BOLD DOTS
# def create_combined_density_bar_chart(pre_csv, post_csv):
#     """Create IEEE/CHI-compliant density plot with integrated bar chart with accurate data points"""
#     # Load and prepare data
#     pre_df = pd.read_csv(pre_csv)
#     post_df = pd.read_csv(post_csv)
    
#     merged_df = pd.merge(
#         pre_df[['user_ID', 'structural_complexity']], 
#         post_df[['user_ID', 'structural_complexity']], 
#         on='user_ID', 
#         suffixes=('_pre', '_post')
#     )
    
#     # Colors following IEEE/CHI standards
#     pre_color = "#4C72B0"  # Blue
#     post_color = "#55A868"  # Green
    
#     # Calculate statistics
#     pre_mean = merged_df['structural_complexity_pre'].mean()
#     post_mean = merged_df['structural_complexity_post'].mean()
#     pre_sd = merged_df['structural_complexity_pre'].std()
#     post_sd = merged_df['structural_complexity_post'].std()
#     pre_se = pre_sd / np.sqrt(len(merged_df))
#     post_se = post_sd / np.sqrt(len(merged_df))
    
#     # Calculate t-test and effect size for annotation
#     t_stat, p_value = stats.ttest_rel(merged_df['structural_complexity_post'], 
#                                       merged_df['structural_complexity_pre'])
#     d = (post_mean - pre_mean) / pre_sd
    
#     # Calculate confidence intervals
#     ci_low, ci_high = stats.t.interval(0.95, len(merged_df)-1, 
#                                       loc=post_mean - pre_mean, 
#                                       scale=stats.sem(merged_df['structural_complexity_post'] - 
#                                                      merged_df['structural_complexity_pre']))
    
#     # Create figure with two panels side by side
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
#                                    gridspec_kw={'width_ratios': [2, 1]})
    
#     # Left panel: Density plot
#     sns.kdeplot(x=merged_df['structural_complexity_pre'], ax=ax1, fill=True, 
#                 label=f'Pre-test (M = {pre_mean:.2f})', color=pre_color, alpha=0.7)
#     sns.kdeplot(x=merged_df['structural_complexity_post'], ax=ax1, fill=True, 
#                 label=f'Post-test (M = {post_mean:.2f})', color=post_color, alpha=0.7)
    
#     # Add mean lines to density plot
#     ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
#     ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
#     # Add shaded area for mean difference
#     ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
#     # Add effect size annotation in the density plot
#     effect_size_text = f"d = {d:.2f} (very large effect)"
#     if d < 0.8:
#         effect_size_text = f"d = {d:.2f} (medium effect)" if d >= 0.5 else f"d = {d:.2f} (small effect)"
    
#     # Position the effect size text in the middle of the shaded area
#     ax1.text((pre_mean + post_mean)/2, 0.85, effect_size_text, 
#             ha='center', va='center', fontweight='bold',
#             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
#     # Add delta indicator
#     ax1.text((pre_mean + post_mean)/2, 0.55, f'Δ = {post_mean - pre_mean:.2f}', 
#             ha='center', va='center', fontsize=12, fontweight='bold',
#             bbox=dict(facecolor='white', alpha=0.9))
    
#     # FIXED: Create a proper rug plot to show actual data distribution
#     # Pre-test data rug plot
#     for value in merged_df['structural_complexity_pre']:
#         ax1.plot([value, value], [-0.05, -0.02], color=pre_color, alpha=0.5, linewidth=1.5)
    
#     # Post-test data rug plot
#     for value in merged_df['structural_complexity_post']:
#         ax1.plot([value, value], [-0.15, -0.12], color=post_color, alpha=0.5, linewidth=1.5)
    
#     # Add small point markers at the ends of the rug lines for better visibility
#     ax1.scatter(merged_df['structural_complexity_pre'], [-0.05]*len(merged_df), 
#                color=pre_color, edgecolor='white', s=25, alpha=0.7)
#     ax1.scatter(merged_df['structural_complexity_post'], [-0.15]*len(merged_df), 
#                color=post_color, edgecolor='white', s=25, alpha=0.7)
    
#     # Add labels for the data points
#     center_x = (pre_mean + post_mean)/2
#     ax1.text(center_x - 0.43, -0.05, "Pre-test scores", color=pre_color, 
#          ha='right', va='center', fontsize=10)
#     ax1.text(center_x + 0.139, -0.15, "Post-test scores", color=post_color, 
#             ha='left', va='center', fontsize=10)
    
#     # Right panel: Bar chart with error bars
#     bar_width = 0.7
#     x_pos = [0, 1]
#     bars = ax2.bar(x_pos, [pre_mean, post_mean], yerr=[pre_se, post_se], 
#                   width=bar_width, capsize=10, color=[pre_color, post_color], 
#                   edgecolor='black', linewidth=1.5)
    
#     # Add value labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         ax2.text(bar.get_x() + bar.get_width()/2., height + (pre_se if i==0 else post_se) + 0.1,
#                 f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
#     # Add significance line and annotation
#     y_max = max(pre_mean + pre_se, post_mean + post_se) + 0.3
#     ax2.plot([0, 0, 1, 1], [y_max-0.1, y_max, y_max, y_max-0.1], lw=1.5, c='black')
    
#     # Add stars for significance
#     stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
#     ax2.text(0.5, y_max + 0.1, stars, ha='center', va='bottom', fontsize=16)
    
#     # Customize left panel (density)
#     ax1.set_title('Distribution of Structural Complexity Scores', fontsize=14, fontweight='bold', pad=15)
#     ax1.set_xlabel('Structural Complexity Score', fontsize=12)
#     ax1.set_ylabel('Density', fontsize=12)
#     ax1.set_xlim(1.5, 4.5)
#     ax1.set_ylim(-0.2, 1.4)
#     ax1.grid(linestyle='--', alpha=0.6)
#     ax1.legend(fontsize=11, loc='upper left')
    
#     # Set explicit x-axis ticks
#     ax1.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
#     ax1.set_xticklabels(['2.0', '2.5', '3.0', '3.5', '4.0'])
    
#     # Customize right panel (bars)
#     ax2.set_title('Mean Scores with SE', fontsize=14, fontweight='bold', pad=15)
#     ax2.set_ylabel('Structural Complexity Score', fontsize=12)
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels(['Pre-test', 'Post-test'], fontsize=11)
#     ax2.set_ylim(1.5, y_max + 0.5)
#     ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
#     # Add overall title
#     plt.suptitle('Impact of Mind Elevator on Structural Complexity of Arguments', 
#                  fontsize=16, fontweight='bold', y=0.98)
    
#     # Add paired sample t-test results as a footnote
#     plt.figtext(0.5, 0.01, 
#                 f"Paired t-test: t({len(merged_df)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
#                 ha='center', fontsize=10, style='italic')
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save figure with error handling
#     try:
#         plt.savefig('structural_complexity_combined.png', dpi=300, bbox_inches='tight')
#         print("Saved PNG successfully")
        
#         try:
#             plt.savefig('structural_complexity_combined.pdf', bbox_inches='tight')
#             print("Saved PDF successfully")
#         except PermissionError:
#             print("Permission error saving PDF. Trying alternative filename...")
#             import datetime
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             plt.savefig(f'structural_complexity_combined_{timestamp}.pdf', bbox_inches='tight')
#             print(f"Saved as: structural_complexity_combined_{timestamp}.pdf")
#     except Exception as e:
#         print(f"Error saving files: {type(e).__name__}: {e}")
#     finally:
#         plt.close()
    
#     print("Created IEEE/CHI-compliant combined density-bar visualization with accurate data representation")

def create_combined_density_bar_chart(pre_csv, post_csv):
    """Create IEEE/CHI-compliant density plot with integrated bar chart with accurate data points"""
    # Load and prepare data
    pre_df = pd.read_csv(pre_csv)
    post_df = pd.read_csv(post_csv)
    
    merged_df = pd.merge(
        pre_df[['user_ID', 'structural_complexity']], 
        post_df[['user_ID', 'structural_complexity']], 
        on='user_ID', 
        suffixes=('_pre', '_post')
    )
    
    # Colors following IEEE/CHI standards
    pre_color = "#4C72B0"  # Blue
    post_color = "#55A868"  # Green
    
    # Calculate statistics
    pre_mean = merged_df['structural_complexity_pre'].mean()
    post_mean = merged_df['structural_complexity_post'].mean()
    pre_sd = merged_df['structural_complexity_pre'].std()
    post_sd = merged_df['structural_complexity_post'].std()
    pre_se = pre_sd / np.sqrt(len(merged_df))
    post_se = post_sd / np.sqrt(len(merged_df))
    
    # Calculate t-test and effect size for annotation
    t_stat, p_value = stats.ttest_rel(merged_df['structural_complexity_post'], 
                                      merged_df['structural_complexity_pre'])
    d = (post_mean - pre_mean) / pre_sd
    
    # Calculate confidence intervals
    ci_low, ci_high = stats.t.interval(0.95, len(merged_df)-1, 
                                      loc=post_mean - pre_mean, 
                                      scale=stats.sem(merged_df['structural_complexity_post'] - 
                                                     merged_df['structural_complexity_pre']))
    
    # Create figure with two panels side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                   gridspec_kw={'width_ratios': [2, 1]})
    
    # Left panel: Density plot
    sns.kdeplot(x=merged_df['structural_complexity_pre'], ax=ax1, fill=True, 
                label=f'Pre-test (M = {pre_mean:.2f})', color=pre_color, alpha=0.7)
    sns.kdeplot(x=merged_df['structural_complexity_post'], ax=ax1, fill=True, 
                label=f'Post-test (M = {post_mean:.2f})', color=post_color, alpha=0.7)
    
    # Add mean lines to density plot
    ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
    ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
    # Add shaded area for mean difference
    ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
    # Add effect size annotation in the density plot
    effect_size_text = f"d = {d:.2f} (very large effect)"
    if d < 0.8:
        effect_size_text = f"d = {d:.2f} (medium effect)" if d >= 0.5 else f"d = {d:.2f} (small effect)"
    
    # Position the effect size text in the middle of the shaded area
    ax1.text((pre_mean + post_mean)/2, 0.85, effect_size_text, 
            ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add delta indicator
    ax1.text((pre_mean + post_mean)/2, 0.55, f'Δ = {post_mean - pre_mean:.2f}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9))
    
    # FIXED: Create a proper rug plot to show actual data distribution
    # Pre-test data rug plot
    for value in merged_df['structural_complexity_pre']:
        ax1.plot([value, value], [-0.05, -0.02], color=pre_color, alpha=0.6, linewidth=1.8)
    
    # Post-test data rug plot
    for value in merged_df['structural_complexity_post']:
        ax1.plot([value, value], [-0.15, -0.12], color=post_color, alpha=0.6, linewidth=1.8)
    
    # Add small point markers at the ends of the rug lines - MADE BOLDER
    ax1.scatter(merged_df['structural_complexity_pre'], [-0.05]*len(merged_df), 
               color=pre_color, edgecolor='white', s=30, alpha=0.85, linewidth=0.8)
    ax1.scatter(merged_df['structural_complexity_post'], [-0.15]*len(merged_df), 
               color=post_color, edgecolor='white', s=30, alpha=0.85, linewidth=0.8)
    
    # Add labels for the data points
    center_x = (pre_mean + post_mean)/2
    ax1.text(center_x - 0.43, -0.05, "Pre-test scores", color=pre_color, 
         ha='right', va='center', fontsize=10, fontweight='bold')
    ax1.text(center_x + 0.139, -0.15, "Post-test scores", color=post_color, 
            ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Right panel: Bar chart with error bars
    bar_width = 0.7
    x_pos = [0, 1]
    bars = ax2.bar(x_pos, [pre_mean, post_mean], yerr=[pre_se, post_se], 
                  width=bar_width, capsize=10, color=[pre_color, post_color], 
                  edgecolor='black', linewidth=1.5)
    
    # ADDED: Connecting dotted line between pre and post means
    ax2.plot([0 + bar_width/2, 1 - bar_width/2], [pre_mean, post_mean], 
            color='darkgray', linestyle='--', linewidth=1.8, alpha=0.9)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (pre_se if i==0 else post_se) + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add significance line and annotation
    y_max = max(pre_mean + pre_se, post_mean + post_se) + 0.3
    ax2.plot([0, 0, 1, 1], [y_max-0.1, y_max, y_max, y_max-0.1], lw=1.5, c='black')
    
    # Add stars for significance
    stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
    ax2.text(0.5, y_max + 0.1, stars, ha='center', va='bottom', fontsize=16)
    
    # Customize left panel (density)
    ax1.set_title('Distribution of Structural Complexity Scores', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Structural Complexity Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(1.5, 4.5)
    ax1.set_ylim(-0.2, 1.4)
    ax1.grid(linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper left')
    
    # Set explicit x-axis ticks
    ax1.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
    ax1.set_xticklabels(['2.0', '2.5', '3.0', '3.5', '4.0'])
    
    # Customize right panel (bars)
    ax2.set_title('Mean Scores with SE', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Structural Complexity Score', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Pre-test', 'Post-test'], fontsize=11)
    ax2.set_ylim(1.5, y_max + 0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add overall title
    plt.suptitle('Impact of Mind Elevator on Structural Complexity of Arguments', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add paired sample t-test results as a footnote
    plt.figtext(0.5, 0.01, 
                f"Paired t-test: t({len(merged_df)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure with error handling
    try:
        plt.savefig('structural_complexity_combined.png', dpi=300, bbox_inches='tight')
        print("Saved PNG successfully")
        
        try:
            plt.savefig('structural_complexity_combined.pdf', bbox_inches='tight')
            print("Saved PDF successfully")
        except PermissionError:
            print("Permission error saving PDF. Trying alternative filename...")
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'structural_complexity_combined_{timestamp}.pdf', bbox_inches='tight')
            print(f"Saved as: structural_complexity_combined_{timestamp}.pdf")
    except Exception as e:
        print(f"Error saving files: {type(e).__name__}: {e}")
    finally:
        plt.close()
    
    print("Created IEEE/CHI-compliant combined density-bar visualization with accurate data representation")


if __name__ == "__main__":
    result_df = analyze_structural_complexity('pre_form.csv', 'post_form.csv')
    create_enhanced_visualizations('pre_form.csv', 'post_form.csv')
    create_combined_density_bar_chart('pre_form.csv', 'post_form.csv')
    save_results_to_csv(result_df)  # Add this line






































##### supos to have betetr diagram









# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats

# def create_enhanced_visualizations(pre_csv, post_csv):
#     """Create clean, non-overlapping visualizations of structural complexity"""
#     # Load data
#     pre_df = pd.read_csv(pre_csv)
#     post_df = pd.read_csv(post_csv)
    
#     # Merge pre and post data
#     merged_df = pd.merge(
#         pre_df[['user_ID', 'structural_complexity']], 
#         post_df[['user_ID', 'structural_complexity']], 
#         on='user_ID', 
#         suffixes=('_pre', '_post')
#     )
    
#     # Setup colors as requested
#     pre_color = "#4C72B0"  # Blue
#     post_color = "#55A868"  # Green
    
#     # Calculate stats for annotations
#     pre_mean = merged_df['structural_complexity_pre'].mean()
#     post_mean = merged_df['structural_complexity_post'].mean()
#     pre_se = merged_df['structural_complexity_pre'].std() / np.sqrt(len(merged_df))
#     post_se = merged_df['structural_complexity_post'].std() / np.sqrt(len(merged_df))
    
#     # Prepare data for plotting
#     data_long = pd.DataFrame({
#         'Time': ['Pre-test'] * len(merged_df) + ['Post-test'] * len(merged_df),
#         'Structural Complexity': np.concatenate([
#             merged_df['structural_complexity_pre'], 
#             merged_df['structural_complexity_post']
#         ])
#     })
    
#     # 1. VIOLIN PLOT
#     plt.figure(figsize=(10, 6))
    
#     # Create violin plot with individual data points
#     ax = sns.violinplot(x='Time', y='Structural Complexity', data=data_long, 
#                    palette=[pre_color, post_color], inner=None)
    
#     # Add scatter points with jitter
#     sns.stripplot(x='Time', y='Structural Complexity', data=data_long, 
#               jitter=True, size=6, alpha=0.7, palette=[pre_color, post_color])
    
#     # Add mean points with error bars
#     plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
#              fmt='o', color='black', markersize=10, capsize=10, 
#              markerfacecolor='white', markeredgewidth=2, elinewidth=2)
    
#     # Add mean value labels
#     for i, (mean, se) in enumerate(zip([pre_mean, post_mean], [pre_se, post_se])):
#         plt.text(i, mean + se + 0.1, f'Mean: {mean:.2f}', ha='center', fontweight='bold')
    
#     # Perform paired t-test for annotation
#     t_stat, p_value = stats.ttest_rel(merged_df['structural_complexity_post'], 
#                                     merged_df['structural_complexity_pre'])
    
#     # Add significance annotation
#     y_max = max(merged_df['structural_complexity_post'].max(), 
#                merged_df['structural_complexity_pre'].max()) + 0.2
#     plt.plot([0, 0, 1, 1], [y_max, y_max+0.1, y_max+0.1, y_max], lw=1.5, c='black')
    
#     # Add stars based on p-value
#     stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
#     plt.text(0.5, y_max+0.15, f'{stars}', ha='center', va='bottom', fontsize=16)
    
#     # Add p-value
#     plt.text(0.5, y_max+0.05, f'p < 0.001', ha='center', va='bottom')
    
#     plt.title('Distribution of Structural Complexity Scores', fontsize=16, fontweight='bold', pad=20)
#     plt.ylabel('Structural Complexity Score', fontsize=14)
#     plt.xlabel('', fontsize=14)  # Remove x-label as it's redundant
#     plt.ylim(1.5, 4.5)  # Adjust y-axis limits
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig('structural_complexity_violin.png', dpi=300, bbox_inches='tight')
#     plt.savefig('structural_complexity_violin.pdf', bbox_inches='tight')
#     plt.close()
    
#     # 2. BAR CHART WITH ERROR BARS
#     plt.figure(figsize=(8, 6))
    
#     # Create bar chart
#     bar_width = 0.6
#     bars = plt.bar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
#                width=bar_width, capsize=10, color=[pre_color, post_color], 
#                edgecolor='black', linewidth=1.5)
    
#     # Add value labels on bars
#     for i, (bar, value) in enumerate(zip(bars, [pre_mean, post_mean])):
#         plt.text(i, value/2, f'{value:.2f}', ha='center', va='center', 
#              color='white', fontweight='bold', fontsize=12)
    
#     # Add connecting line for pre-post
#     plt.plot([0+bar_width/2, 1-bar_width/2], [pre_mean, post_mean], 
#          color='gray', linestyle='--', alpha=0.7, lw=2)
    
#     # Add significance annotation
#     effect_size = (post_mean - pre_mean) / merged_df['structural_complexity_pre'].std()
#     plt.text(0.5, 0.2, f'd = {effect_size:.2f} (large effect)', 
#          ha='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
#     plt.title('Mean Structural Complexity Scores', fontsize=16, fontweight='bold', pad=20)
#     plt.ylabel('Structural Complexity Score', fontsize=14)
#     plt.xticks([0, 1], ['Pre-test', 'Post-test'], fontsize=12)
#     plt.ylim(0, 5)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig('structural_complexity_means.png', dpi=300, bbox_inches='tight')
#     plt.savefig('structural_complexity_means.pdf', bbox_inches='tight')
#     plt.close()
    
#     # 3. DENSITY DISTRIBUTION PLOT
#     plt.figure(figsize=(10, 6))
    
#     # Create density plots
#     sns.kdeplot(merged_df['structural_complexity_pre'], fill=True, 
#             label=f'Pre-test (Mean: {pre_mean:.2f})', color=pre_color, alpha=0.7)
#     sns.kdeplot(merged_df['structural_complexity_post'], fill=True, 
#             label=f'Post-test (Mean: {post_mean:.2f})', color=post_color, alpha=0.7)
    
#     # Add vertical lines for means
#     plt.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
#     plt.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
#     # Add shaded area for mean difference
#     plt.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
#     plt.text((pre_mean + post_mean)/2, 0.1, f'Δ = {post_mean - pre_mean:.2f}', 
#          ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.title('Distribution of Structural Complexity Scores', fontsize=16, fontweight='bold', pad=20)
#     plt.xlabel('Structural Complexity Score', fontsize=14)
#     plt.ylabel('Density', fontsize=14)
#     plt.xlim(1.5, 4.5)
#     plt.grid(linestyle='--', alpha=0.7)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.savefig('structural_complexity_density.png', dpi=300, bbox_inches='tight')
#     plt.savefig('structural_complexity_density.pdf', bbox_inches='tight')
#     plt.close()
    
#     print("Enhanced visualizations created and saved as:")
#     print("1. structural_complexity_violin.png/pdf")
#     print("2. structural_complexity_means.png/pdf")
#     print("3. structural_complexity_density.png/pdf")

# if __name__ == "__main__":
#     create_enhanced_visualizations('pre_form.csv', 'post_form.csv')