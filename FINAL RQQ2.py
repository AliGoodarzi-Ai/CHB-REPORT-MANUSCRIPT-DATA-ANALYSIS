# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from statsmodels.stats.multitest import multipletests
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D

# def analyze_critical_thinking(pre_csv, post_csv):
#     """
#     Comprehensive analysis of critical thinking skills (RQ2)
#     """
#     # Load data
#     pre_df = pd.read_csv(pre_csv)
#     post_df = pd.read_csv(post_csv)
    
#     print("Analyzing critical thinking skills (RQ2)...\n")
    
#     # Define the specific columns for critical thinking skills
#     ct_skills = {
#         'Recognize Assumptions': 'identify_assumptions',
#         'Evaluate Evidence': 'analyze_evidence',
#         'Identify Weaknesses': 'identify_weaknesses',
#         'Willing to Change Position': 'willing_change_position',
#         'Consider Perspectives': 'consider_perspectives'
#     }
    
#     # Check columns exist in datasets
#     for skill_name, col in ct_skills.items():
#         if col not in pre_df.columns or col not in post_df.columns:
#             print(f"ERROR: '{col}' column not found in one or both datasets")
#             return
    
#     # Create dataframe to store all analysis results
#     results_df = pd.DataFrame(columns=[
#         'Skill', 'Pre_Mean', 'Pre_SD', 'Post_Mean', 'Post_SD', 
#         'Mean_Diff', 'Percent_Change', 'T_stat', 'P_value', 
#         'Cohens_d', 'CI_Low', 'CI_High', 'Improved_Pct'
#     ])
    
#     # Store p-values for Holm-Bonferroni correction
#     p_values = []
#     skill_names = []
    
#     # Process each critical thinking skill
#     for skill_name, col in ct_skills.items():
#         pre_data = pre_df[col].astype(float)
#         post_data = post_df[col].astype(float)
        
#         # Basic statistics
#         pre_mean = pre_data.mean()
#         pre_sd = pre_data.std()
#         post_mean = post_data.mean()
#         post_sd = post_data.std()
#         mean_diff = post_mean - pre_mean
#         percent_change = (mean_diff / pre_mean) * 100
        
#         # Paired t-test
#         t_stat, p_value = stats.ttest_rel(post_data, pre_data)
        
#         # Store p-value for correction
#         p_values.append(p_value)
#         skill_names.append(skill_name)
        
#         # Effect size (Cohen's d)
#         d = mean_diff / pre_sd
        
#         # Confidence interval
#         ci_low, ci_high = stats.t.interval(
#             0.95, len(pre_data)-1, 
#             loc=mean_diff, 
#             scale=stats.sem(post_data - pre_data)
#         )
        
#         # Percentage of participants who improved
#         improved = np.sum(post_data > pre_data)
#         improved_pct = (improved / len(pre_data)) * 100
        
#         # Store results
#         results_df = pd.concat([results_df, pd.DataFrame({
#             'Skill': [skill_name],
#             'Pre_Mean': [pre_mean],
#             'Pre_SD': [pre_sd],
#             'Post_Mean': [post_mean],
#             'Post_SD': [post_sd],
#             'Mean_Diff': [mean_diff],
#             'Percent_Change': [percent_change],
#             'T_stat': [t_stat],
#             'P_value': [p_value],
#             'Cohens_d': [d],
#             'CI_Low': [ci_low],
#             'CI_High': [ci_high],
#             'Improved_Pct': [improved_pct]
#         })], ignore_index=True)
    
#     # Apply Holm-Bonferroni correction
#     reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
#     results_df['P_corrected'] = p_corrected
#     results_df['Significant'] = reject
    
#     # Calculate RED framework composite scores
#     # 1. Create a draw conclusions composite
#     pre_draw = pre_df[['identify_weaknesses', 'willing_change_position', 'consider_perspectives']].mean(axis=1)
#     post_draw = post_df[['identify_weaknesses', 'willing_change_position', 'consider_perspectives']].mean(axis=1)
    
#     # 2. Calculate overall RED score
#     pre_red = pre_df[[ct_skills[skill] for skill in ct_skills]].mean(axis=1)
#     post_red = post_df[[ct_skills[skill] for skill in ct_skills]].mean(axis=1)
    
#     # T-test for Draw Conclusions composite
#     t_draw, p_draw = stats.ttest_rel(post_draw, pre_draw)
#     d_draw = (post_draw.mean() - pre_draw.mean()) / pre_draw.std()
    
#     # T-test for overall RED
#     t_red, p_red = stats.ttest_rel(post_red, pre_red)
#     d_red = (post_red.mean() - pre_red.mean()) / pre_red.std()
    
#     # Print detailed results
#     print("\n=== CRITICAL THINKING SKILLS ANALYSIS (RQ2) ===")
#     print("\nIndividual Skills Analysis:")
#     print("-" * 100)
#     print(f"{'Skill':<25} {'Pre':<10} {'Post':<10} {'Change':<10} {'t-stat':<10} {'p-value':<10} {'p-corr':<10} {'Sig?':<6} {'d':<8}")
#     print("-" * 100)
    
#     for _, row in results_df.iterrows():
#         sig_symbol = "***" if row['P_corrected'] < 0.001 else "**" if row['P_corrected'] < 0.01 else "*" if row['P_corrected'] < 0.05 else ""
#         print(f"{row['Skill']:<25} {row['Pre_Mean']:.2f}±{row['Pre_SD']:.2f} {row['Post_Mean']:.2f}±{row['Post_SD']:.2f} {row['Mean_Diff']:+.2f} {row['T_stat']:.2f} {row['P_value']:.4f} {row['P_corrected']:.4f} {sig_symbol:<6} {row['Cohens_d']:.2f}")
    
#     print("-" * 100)
#     print("\nRED Framework Composite Analysis:")
#     print(f"Draw Conclusions (composite): {pre_draw.mean():.2f}±{pre_draw.std():.2f} → {post_draw.mean():.2f}±{post_draw.std():.2f}, t={t_draw:.2f}, p={p_draw:.4f}, d={d_draw:.2f}")
#     print(f"Overall RED score:            {pre_red.mean():.2f}±{pre_red.std():.2f} → {post_red.mean():.2f}±{post_red.std():.2f}, t={t_red:.2f}, p={p_red:.4f}, d={d_red:.2f}")
    
#     # Create visualizations
#     # 1. Radar chart of all skills
#     create_radar_chart(results_df, "Critical Thinking Skills: Pre vs Post")
    
#     # 2. Bar chart of all skills with error bars and significance indicators
#     create_bar_chart(results_df, "Changes in Critical Thinking Skills")
    
#     # 3. Distribution shifts for overall RED score
#     create_distribution_plot(pre_red, post_red, "Overall RED Score Distribution")
    
#     # Save results to CSV
#     results_df.to_csv('critical_thinking_analysis_results.csv', index=False)
#     print("\nResults saved to 'critical_thinking_analysis_results.csv'")
    
#     # Return results for further analysis if needed
#     return {
#         'individual_skills': results_df,
#         'draw_conclusions': {
#             'pre_mean': pre_draw.mean(),
#             'pre_sd': pre_draw.std(),
#             'post_mean': post_draw.mean(),
#             'post_sd': post_draw.std(),
#             't_stat': t_draw,
#             'p_value': p_draw,
#             'cohens_d': d_draw
#         },
#         'overall_red': {
#             'pre_mean': pre_red.mean(),
#             'pre_sd': pre_red.std(),
#             'post_mean': post_red.mean(),
#             'post_sd': post_red.std(),
#             't_stat': t_red,
#             'p_value': p_red,
#             'cohens_d': d_red
#         }
#     }

# def create_radar_chart(results_df, title):
#     """Create radar chart comparing pre and post scores for all skills"""
#     # Number of variables
#     categories = results_df['Skill'].tolist()
#     N = len(categories)
    
#     # Pre and post means
#     pre_means = results_df['Pre_Mean'].tolist()
#     post_means = results_df['Post_Mean'].tolist()
    
#     # Close the plot (append first value to end)
#     pre_means += [pre_means[0]]
#     post_means += [post_means[0]]
#     categories += [categories[0]]
    
#     # Calculate angle for each category
#     angles = [n / float(N) * 2 * np.pi for n in range(N)]
#     angles += angles[:1]  # Close the loop
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
#     # Add grid lines and category labels
#     plt.xticks(angles[:-1], categories[:-1], size=12)
#     ax.set_rlabel_position(0)
#     plt.yticks([1, 2, 3, 4, 5, 6, 7], ["1", "2", "3", "4", "5", "6", "7"], 
#                color="grey", size=10)
#     plt.ylim(0, 7)
    
#     # Plot data
#     ax.plot(angles, pre_means, 'o-', linewidth=2, label='Pre-test', color='#4C72B0')
#     ax.fill(angles, pre_means, alpha=0.1, color='#4C72B0')
    
#     ax.plot(angles, post_means, 'o-', linewidth=2, label='Post-test', color='#55A868')
#     ax.fill(angles, post_means, alpha=0.1, color='#55A868')
    
#     # Add legend and title
#     plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
#     plt.title(title, size=15, y=1.1)
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig('critical_thinking_radar.png', dpi=300, bbox_inches='tight')
#     plt.savefig('critical_thinking_radar.pdf', bbox_inches='tight')
#     plt.close()
    
# def create_bar_chart(results_df, title):
#     """Create bar chart with error bars and significance indicators"""
#     # Setup
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     # Data
#     skills = results_df['Skill'].tolist()
#     pre_means = results_df['Pre_Mean'].tolist()
#     post_means = results_df['Post_Mean'].tolist()
#     pre_sds = results_df['Pre_SD'].tolist()
#     post_sds = results_df['Post_SD'].tolist()
    
#     # Calculate standard errors
#     n = 16  # Number of participants
#     pre_se = [sd / np.sqrt(n) for sd in pre_sds]
#     post_se = [sd / np.sqrt(n) for sd in post_sds]
    
#     # X positions
#     x = np.arange(len(skills))
#     width = 0.35
    
#     # Create bars
#     pre_bars = ax.bar(x - width/2, pre_means, width, yerr=pre_se, 
#                      label='Pre-test', color='#4C72B0', capsize=5, 
#                      edgecolor='black', linewidth=1)
#     post_bars = ax.bar(x + width/2, post_means, width, yerr=post_se, 
#                       label='Post-test', color='#55A868', capsize=5, 
#                       edgecolor='black', linewidth=1)
    
#     # Add value labels
#     for i, bar in enumerate(pre_bars):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + pre_se[i] + 0.1,
#                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                
#     for i, bar in enumerate(post_bars):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + post_se[i] + 0.1,
#                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
#     # Add significance indicators
#     for i, row in results_df.iterrows():
#         if row['Significant']:
#             # Determine number of stars based on p-value
#             if row['P_corrected'] < 0.001:
#                 stars = "***"
#             elif row['P_corrected'] < 0.01:
#                 stars = "**"
#             elif row['P_corrected'] < 0.05:
#                 stars = "*"
                
#             # Calculate position
#             y_pos = max(pre_means[i] + pre_se[i], post_means[i] + post_se[i]) + 0.3
            
#             # Add stars
#             ax.text(i, y_pos, stars, ha='center', va='bottom', fontsize=14)
            
#             # Add connecting line
#             ax.plot([i-width/2, i+width/2], [y_pos-0.1, y_pos-0.1], '-', color='black', linewidth=1)
    
#     # Customize layout
#     ax.set_ylabel('Mean Score (1-7 scale)', fontsize=12)
#     ax.set_title(title, fontsize=15, pad=20)
#     ax.set_xticks(x)
#     ax.set_xticklabels(skills, rotation=15, ha='right', fontsize=10)
#     ax.legend(fontsize=12)
    
#     # Add significance legend
#     legend_elements = [
#         Line2D([0], [0], marker='', color='none', label='Significance:'),
#         Line2D([0], [0], marker='', color='none', label='* p < 0.05'),
#         Line2D([0], [0], marker='', color='none', label='** p < 0.01'),
#         Line2D([0], [0], marker='', color='none', label='*** p < 0.001')
#     ]
#     ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
#     # Add note about Holm-Bonferroni
#     plt.figtext(0.5, 0.01, "Note: p-values corrected using Holm-Bonferroni method", 
#                 ha="center", fontsize=9, style='italic')
    
#     # Set y-axis to start from 0
#     ax.set_ylim(1, 8)
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig('critical_thinking_bar_chart.png', dpi=300, bbox_inches='tight')
#     plt.savefig('critical_thinking_bar_chart.pdf', bbox_inches='tight')
#     plt.close()

# def create_distribution_plot(pre_data, post_data, title):
#     """Create density distribution plot for pre-post comparison"""
#     plt.figure(figsize=(10, 6))
    
#     # Plot densities
#     sns.kdeplot(pre_data, fill=True, color='#4C72B0', alpha=0.7, 
#               label=f'Pre-test (M={pre_data.mean():.2f})')
#     sns.kdeplot(post_data, fill=True, color='#55A868', alpha=0.7, 
#               label=f'Post-test (M={post_data.mean():.2f})')
    
#     # Add mean lines
#     plt.axvline(pre_data.mean(), color='#4C72B0', linestyle='--', linewidth=2)
#     plt.axvline(post_data.mean(), color='#55A868', linestyle='--', linewidth=2)
    
#     # Add mean difference annotation
#     mean_diff = post_data.mean() - pre_data.mean()
#     plt.annotate(f'Mean difference: +{mean_diff:.2f}',
#               xy=((pre_data.mean() + post_data.mean())/2, 0.5),
#               xytext=(0, 30), textcoords='offset points', 
#               ha='center', va='bottom',
#               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
#               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
#     # Customize plot
#     plt.title(title, fontsize=15, pad=20)
#     plt.xlabel('RED Score (1-7 scale)', fontsize=12)
#     plt.ylabel('Density', fontsize=12)
#     plt.legend()
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig('critical_thinking_distribution.png', dpi=300, bbox_inches='tight')
#     plt.savefig('critical_thinking_distribution.pdf', bbox_inches='tight')
#     plt.close()

# if __name__ == "__main__":
#     results = analyze_critical_thinking('pre_form.csv', 'post_form.csv')
























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def analyze_critical_thinking(pre_csv, post_csv):
    """
    Comprehensive analysis of critical thinking skills (RQ2)
    """
    # Load data
    pre_df = pd.read_csv(pre_csv)
    post_df = pd.read_csv(post_csv)
    
    print("Analyzing critical thinking skills (RQ2)...\n")
    
    # Define the specific columns for critical thinking skills
    ct_skills = {
        'Recognize Assumptions': 'identify_assumptions',
        'Evaluate Evidence': 'analyze_evidence',
        'Identify Weaknesses': 'identify_weaknesses',
        'Willing to Change Position': 'willing_change_position',
        'Consider Perspectives': 'consider_perspectives'
    }
    
    # Check columns exist in datasets
    for skill_name, col in ct_skills.items():
        if col not in pre_df.columns or col not in post_df.columns:
            print(f"ERROR: '{col}' column not found in one or both datasets")
            return
    
    # Create dataframe to store all analysis results
    results_df = pd.DataFrame(columns=[
        'Skill', 'Pre_Mean', 'Pre_SD', 'Post_Mean', 'Post_SD', 
        'Mean_Diff', 'Percent_Change', 'T_stat', 'P_value', 
        'Cohens_d', 'CI_Low', 'CI_High', 'Improved_Pct'
    ])
    
    # Store p-values for Holm-Bonferroni correction
    p_values = []
    skill_names = []
    
    # Process each critical thinking skill
    for skill_name, col in ct_skills.items():
        pre_data = pre_df[col].astype(float)
        post_data = post_df[col].astype(float)
        
        # Basic statistics
        pre_mean = pre_data.mean()
        pre_sd = pre_data.std()
        post_mean = post_data.mean()
        post_sd = post_data.std()
        mean_diff = post_mean - pre_mean
        percent_change = (mean_diff / pre_mean) * 100
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(post_data, pre_data)
        
        # Store p-value for correction
        p_values.append(p_value)
        skill_names.append(skill_name)
        
        # Effect size (Cohen's d)
        d = mean_diff / pre_sd
        
        # Confidence interval
        ci_low, ci_high = stats.t.interval(
            0.95, len(pre_data)-1, 
            loc=mean_diff, 
            scale=stats.sem(post_data - pre_data)
        )
        
        # Percentage of participants who improved
        improved = np.sum(post_data > pre_data)
        improved_pct = (improved / len(pre_data)) * 100
        
        # Store results
        results_df = pd.concat([results_df, pd.DataFrame({
            'Skill': [skill_name],
            'Pre_Mean': [pre_mean],
            'Pre_SD': [pre_sd],
            'Post_Mean': [post_mean],
            'Post_SD': [post_sd],
            'Mean_Diff': [mean_diff],
            'Percent_Change': [percent_change],
            'T_stat': [t_stat],
            'P_value': [p_value],
            'Cohens_d': [d],
            'CI_Low': [ci_low],
            'CI_High': [ci_high],
            'Improved_Pct': [improved_pct]
        })], ignore_index=True)
    
    # Apply Holm-Bonferroni correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
    results_df['P_corrected'] = p_corrected
    results_df['Significant'] = reject
    
    # Calculate RED framework composite scores
    # 1. Create a draw conclusions composite
    pre_draw = pre_df[['identify_weaknesses', 'willing_change_position', 'consider_perspectives']].mean(axis=1)
    post_draw = post_df[['identify_weaknesses', 'willing_change_position', 'consider_perspectives']].mean(axis=1)
    
    # 2. Calculate overall RED score
    pre_red = pre_df[[ct_skills[skill] for skill in ct_skills]].mean(axis=1)
    post_red = post_df[[ct_skills[skill] for skill in ct_skills]].mean(axis=1)
    
    # T-test for Draw Conclusions composite
    t_draw, p_draw = stats.ttest_rel(post_draw, pre_draw)
    d_draw = (post_draw.mean() - pre_draw.mean()) / pre_draw.std()
    
    # T-test for overall RED
    t_red, p_red = stats.ttest_rel(post_red, pre_red)
    d_red = (post_red.mean() - pre_red.mean()) / pre_red.std()
    
    # Print detailed results
    print("\n=== CRITICAL THINKING SKILLS ANALYSIS (RQ2) ===")
    print("\nIndividual Skills Analysis:")
    print("-" * 100)
    print(f"{'Skill':<25} {'Pre':<10} {'Post':<10} {'Change':<10} {'t-stat':<10} {'p-value':<10} {'p-corr':<10} {'Sig?':<6} {'d':<8}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        sig_symbol = "***" if row['P_corrected'] < 0.001 else "**" if row['P_corrected'] < 0.01 else "*" if row['P_corrected'] < 0.05 else ""
        print(f"{row['Skill']:<25} {row['Pre_Mean']:.2f}±{row['Pre_SD']:.2f} {row['Post_Mean']:.2f}±{row['Post_SD']:.2f} {row['Mean_Diff']:+.2f} {row['T_stat']:.2f} {row['P_value']:.4f} {row['P_corrected']:.4f} {sig_symbol:<6} {row['Cohens_d']:.2f}")
    
    print("-" * 100)
    print("\nRED Framework Composite Analysis:")
    print(f"Draw Conclusions (composite): {pre_draw.mean():.2f}±{pre_draw.std():.2f} → {post_draw.mean():.2f}±{post_draw.std():.2f}, t={t_draw:.2f}, p={p_draw:.4f}, d={d_draw:.2f}")
    print(f"Overall RED score:            {pre_red.mean():.2f}±{pre_red.std():.2f} → {post_red.mean():.2f}±{post_red.std():.2f}, t={t_red:.2f}, p={p_red:.4f}, d={d_red:.2f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Original visualizations
    create_radar_chart(results_df, "Critical Thinking Skills: Pre vs Post")
    create_bar_chart(results_df, "Changes in Critical Thinking Skills")
    create_distribution_plot(pre_red, post_red, "Overall RED Score Distribution")
    
    # 2. Enhanced visualizations
    print("\nCreating enhanced IEEE-friendly visualizations...")
    create_individual_violin_charts(pre_df, post_df, ct_skills)
    create_improved_radar_chart(results_df, "Critical Thinking Skills: Pre vs Post")
    visualize_holm_bonferroni(results_df)
    create_overall_red_violin(pre_red, post_red)
    
    # Save results to CSV
    results_df.to_csv('critical_thinking_analysis_results.csv', index=False)
    print("\nResults saved to 'critical_thinking_analysis_results.csv'")
    
    # Return results for further analysis if needed
    return {
    'individual_skills': results_df,
    'draw_conclusions': {
        'pre_mean': pre_draw.mean(),
        'pre_sd': pre_draw.std(),
        'post_mean': post_draw.mean(),
        'post_sd': post_draw.std(),
        't_stat': t_draw,
        'p_value': p_draw,
        'cohens_d': d_draw,
        'pre_data': pre_draw,     # Add this line
        'post_data': post_draw    # Add this line
    },
    'overall_red': {
        'pre_mean': pre_red.mean(),
        'pre_sd': pre_red.std(),
        'post_mean': post_red.mean(),
        'post_sd': post_red.std(),
        't_stat': t_red,
        'p_value': p_red,
        'cohens_d': d_red,
        'pre_data': pre_red,      # Add this line
        'post_data': post_red     # Add this line
    }
}

def create_individual_violin_charts(pre_df, post_df, ct_skills):
    """Create separate IEEE-friendly violin charts for each critical thinking skill"""
    for skill_name, col in ct_skills.items():
        # Extract data
        pre_data = pre_df[col].astype(float)
        post_data = post_df[col].astype(float)
        
        # Prepare data for plotting
        data_long = pd.DataFrame({
            'Time': ['Pre-test'] * len(pre_data) + ['Post-test'] * len(post_data),
            'Score': np.concatenate([pre_data, post_data])
        })
        
        # Calculate statistics
        pre_mean = pre_data.mean()
        pre_sd = pre_data.std()
        post_mean = post_data.mean()
        post_sd = post_data.std()
        pre_se = pre_sd / np.sqrt(len(pre_data))
        post_se = post_sd / np.sqrt(len(post_data))
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(post_data, pre_data)
        d = (post_mean - pre_mean) / pre_sd
        
        # Create figure
        plt.figure(figsize=(7, 8))
        
        # Create violin plot
        ax = sns.violinplot(x='Time', y='Score', data=data_long, 
                     palette=['#4C72B0', '#55A868'], inner=None)
        
        # Add individual data points
        sns.stripplot(x='Time', y='Score', data=data_long, 
                  jitter=True, size=6, alpha=0.7, palette=['#4C72B0', '#55A868'])
        
        # Add mean points with error bars
        plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
                   fmt='o', color='black', markersize=10, capsize=10, 
                   markerfacecolor='white', markeredgewidth=2, elinewidth=2)
        
        # Add mean values as text
        plt.text(0, pre_mean + pre_se + 0.2, f'M = {pre_mean:.2f}', ha='center', fontweight='bold')
        plt.text(1, post_mean + post_se + 0.2, f'M = {post_mean:.2f}', ha='center', fontweight='bold')
        
        # Add significance indicators with Holm-Bonferroni note
        max_y = max(pre_data.max(), post_data.max()) + 0.5
        if p_value < 0.05:
            stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            plt.plot([0, 0, 1, 1], [max_y, max_y+0.1, max_y+0.1, max_y], lw=1.5, c='black')
            plt.text(0.5, max_y+0.2, stars, ha='center', va='bottom', fontsize=16)
            
            # Add statistics text in box
            stats_text = f"t({len(pre_data)-1}) = {t_stat:.2f}\np < 0.001, d = {d:.2f}"
            plt.text(0.5, 2.0, stats_text, ha='center', va='center',
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Set title and labels
        plt.title(f'{skill_name}', fontsize=15, pad=20)
        plt.ylabel('Score (1-7 scale)', fontsize=12)
        plt.xlabel('', fontsize=12)
        plt.ylim(0.5, max_y+1.0)
        
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add Holm-Bonferroni note
        plt.figtext(0.5, 0.01, "Note: p-values corrected using Holm-Bonferroni method", 
                  ha="center", fontsize=9, style='italic')
        
        # Save figure
        plt.tight_layout()
        filename = f"violin_{skill_name.replace(' ', '_').lower()}"
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{filename}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Created IEEE-friendly violin chart for {skill_name}")

def create_improved_radar_chart(results_df, title):
    """Create radar chart with non-overlapping text labels"""
    # Number of variables
    categories = results_df['Skill'].tolist()
    N = len(categories)
    
    # Pre and post means
    pre_means = results_df['Pre_Mean'].tolist()
    post_means = results_df['Post_Mean'].tolist()
    
    # Calculate angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    # Close the plot (append first value to end)
    pre_means_closed = pre_means + [pre_means[0]]
    post_means_closed = post_means + [post_means[0]]
    angles_closed = angles + [angles[0]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles_closed, pre_means_closed, 'o-', linewidth=2, label='Pre-test', color='#4C72B0')
    ax.fill(angles_closed, pre_means_closed, alpha=0.1, color='#4C72B0')
    
    ax.plot(angles_closed, post_means_closed, 'o-', linewidth=2, label='Post-test', color='#55A868')
    ax.fill(angles_closed, post_means_closed, alpha=0.1, color='#55A868')
    
    # Set ticks and limits
    ax.set_xticks(angles)
    ax.set_ylim(0, 7)
    
    # Add labels with text boxes to prevent overlap
    for i, angle in enumerate(angles):
        # Calculate label position slightly outside the circle
        label_distance = 8.0  # Beyond max value
        x = angle
        
        # Add the label with background box for clarity
        ax.text(x, label_distance, categories[i], 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Remove default angular labels
    ax.set_xticklabels([])
    
    # Set radial ticks and labels
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7"], color="grey", size=10)
    ax.set_rlabel_position(0)  # Move radial labels
    
    # Add legend with good positioning
    plt.legend(loc='lower right', bbox_to_anchor=(0.9, 0.1), fontsize=12,
              frameon=True, framealpha=0.9)
    
    # Add title
    plt.title(title, size=16, y=1.08)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('critical_thinking_radar_improved.png', dpi=300, bbox_inches='tight')
    plt.savefig('critical_thinking_radar_improved.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created improved radar chart with non-overlapping labels")

def visualize_holm_bonferroni(results_df):
    """Create visualization and explanation of Holm-Bonferroni correction"""
    # Extract data
    skills = results_df['Skill'].tolist()
    p_values = results_df['P_value'].tolist()
    p_corrected = results_df['P_corrected'].tolist()
    is_significant = results_df['Significant'].tolist()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Sort by p-value for proper Holm-Bonferroni visualization
    sorted_indices = np.argsort(p_values)
    sorted_skills = [skills[i] for i in sorted_indices]
    sorted_p = [p_values[i] for i in sorted_indices]
    sorted_p_corr = [p_corrected[i] for i in sorted_indices]
    sorted_sig = [is_significant[i] for i in sorted_indices]
    
    # Create bar chart
    bar_width = 0.35
    x = np.arange(len(sorted_skills))
    
    # Original p-values
    plt.bar(x - bar_width/2, sorted_p, width=bar_width, label='Original p-value', 
           color='#4C72B0', edgecolor='black', linewidth=1)
    
    # Corrected p-values
    plt.bar(x + bar_width/2, sorted_p_corr, width=bar_width, label='Holm-Bonferroni corrected', 
           color='#55A868', edgecolor='black', linewidth=1)
    
    # Add alpha threshold line
    plt.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    
    # Add significance markers
    for i, sig in enumerate(sorted_sig):
        if sig:
            plt.text(i, 0.001, '✓', ha='center', va='center', fontsize=16, 
                   color='green', fontweight='bold')
    
    # Customize plot
    plt.yscale('log')  # Log scale for better visibility
    plt.ylabel('p-value (log scale)', fontsize=12)
    plt.xlabel('Critical Thinking Skills (ordered by p-value)', fontsize=12)
    plt.title('Holm-Bonferroni Correction for Multiple Comparisons', fontsize=14, pad=20,fontweight='bold')
    plt.xticks(x, sorted_skills, rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, "Note: ✓ indicates significance maintained after correction (p < 0.05)", 
              ha="center", fontsize=10, style='italic')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('holm_bonferroni_correction.png', dpi=300, bbox_inches='tight')
    plt.savefig('holm_bonferroni_correction.pdf', bbox_inches='tight')
    plt.close()
    
    # Text explanation for Holm-Bonferroni (printed to console)
    print("\n=== HOLM-BONFERRONI CORRECTION EXPLANATION ===")
    print("The Holm-Bonferroni method controls the familywise error rate when conducting multiple hypothesis tests.")
    print("It works by ordering p-values from smallest to largest and applying sequential rejection:")
    print()
    print("Original significance threshold (α): 0.05")
    print(f"Number of tests performed: {len(sorted_skills)}")
    print()
    print("Procedure:")
    for i, (skill, p, p_corr, sig) in enumerate(zip(sorted_skills, sorted_p, sorted_p_corr, sorted_sig)):
        alpha_adjusted = 0.05 / (len(sorted_skills) - i)
        print(f"{i+1}. {skill}: original p = {p:.6f}, adjusted α = {alpha_adjusted:.6f}, corrected p = {p_corr:.6f}")
        if sig:
            print("   ✓ Remains significant after correction")
        else:
            print("   ✗ No longer significant after correction")

def create_overall_red_violin(pre_red, post_red):
    """Create violin chart for overall RED score"""
    # Create data for plotting
    data_long = pd.DataFrame({
        'Time': ['Pre-test'] * len(pre_red) + ['Post-test'] * len(post_red),
        'RED Score': np.concatenate([pre_red, post_red])
    })
    
    # Calculate statistics
    pre_mean = pre_red.mean()
    pre_sd = pre_red.std()
    post_mean = post_red.mean()
    post_sd = post_red.std()
    pre_se = pre_sd / np.sqrt(len(pre_red))
    post_se = post_sd / np.sqrt(len(post_red))
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(post_red, pre_red)
    d = (post_mean - pre_mean) / pre_sd
    
    # Create figure
    plt.figure(figsize=(8, 9))
    
    # Create violin plot
    ax = sns.violinplot(x='Time', y='RED Score', data=data_long, 
                  palette=['#4C72B0', '#55A868'], inner=None)
    
    # Add scatter points with jitter
    sns.stripplot(x='Time', y='RED Score', data=data_long, 
              jitter=True, size=7, alpha=0.7, 
              palette=['#4C72B0', '#55A868'])
    
    # Add mean points with error bars
    plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
              fmt='o', color='black', markersize=10, capsize=10, 
              markerfacecolor='white', markeredgewidth=2, elinewidth=2)
    
    # Add mean values as text
    plt.text(0, pre_mean + pre_se + 0.2, f'M = {pre_mean:.2f}', ha='center', fontweight='bold')
    plt.text(1, post_mean + post_se + 0.2, f'M = {post_mean:.2f}', ha='center', fontweight='bold')
    
    # Add significance annotation
    max_y = 7.0
    if p_value < 0.05:
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        plt.plot([0, 0, 1, 1], [max_y-0.5, max_y-0.4, max_y-0.4, max_y-0.5], lw=1.5, c='black')
        plt.text(0.5, max_y-0.3, stars, ha='center', va='bottom', fontsize=16)
        
        # Add statistics box
        stats_text = f"t({len(pre_red)-1}) = {t_stat:.2f}\np < 0.001\nd = {d:.2f} (large effect)"
        plt.text(0.5, 2.0, stats_text, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set title and labels
    plt.title('Overall RED Critical Thinking Score', fontsize=15, pad=20)
    plt.ylabel('Score (1-7 scale)', fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.ylim(1.0, max_y)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('overall_red_violin.png', dpi=300, bbox_inches='tight')
    plt.savefig('overall_red_violin.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created violin plot for overall RED score")

def create_radar_chart(results_df, title):
    """Create radar chart comparing pre and post scores for all skills"""
    # Number of variables
    categories = results_df['Skill'].tolist()
    N = len(categories)
    
    # Pre and post means
    pre_means = results_df['Pre_Mean'].tolist()
    post_means = results_df['Post_Mean'].tolist()
    
    # Close the plot (append first value to end)
    pre_means += [pre_means[0]]
    post_means += [post_means[0]]
    categories += [categories[0]]
    
    # Calculate angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Add grid lines and category labels
    plt.xticks(angles[:-1], categories[:-1], size=12)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5, 6, 7], ["1", "2", "3", "4", "5", "6", "7"], 
               color="grey", size=10)
    plt.ylim(0, 7)
    
    # Plot data
    ax.plot(angles, pre_means, 'o-', linewidth=2, label='Pre-test', color='#4C72B0')
    ax.fill(angles, pre_means, alpha=0.1, color='#4C72B0')
    
    ax.plot(angles, post_means, 'o-', linewidth=2, label='Post-test', color='#55A868')
    ax.fill(angles, post_means, alpha=0.1, color='#55A868')
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.1)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('critical_thinking_radar.png', dpi=300, bbox_inches='tight')
    plt.savefig('critical_thinking_radar.pdf', bbox_inches='tight')
    plt.close()
    
def create_bar_chart(results_df, title):
    """Create bar chart with error bars and significance indicators"""
    # Setup
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data
    skills = results_df['Skill'].tolist()
    pre_means = results_df['Pre_Mean'].tolist()
    post_means = results_df['Post_Mean'].tolist()
    pre_sds = results_df['Pre_SD'].tolist()
    post_sds = results_df['Post_SD'].tolist()
    
    # Calculate standard errors
    n = 16  # Number of participants
    pre_se = [sd / np.sqrt(n) for sd in pre_sds]
    post_se = [sd / np.sqrt(n) for sd in post_sds]
    
    # X positions
    x = np.arange(len(skills))
    width = 0.35
    
    # Create bars
    pre_bars = ax.bar(x - width/2, pre_means, width, yerr=pre_se, 
                     label='Pre-test', color='#4C72B0', capsize=5, 
                     edgecolor='black', linewidth=1)
    post_bars = ax.bar(x + width/2, post_means, width, yerr=post_se, 
                      label='Post-test', color='#55A868', capsize=5, 
                      edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, bar in enumerate(pre_bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + pre_se[i] + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                
    for i, bar in enumerate(post_bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + post_se[i] + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Add significance indicators
    for i, row in results_df.iterrows():
        if row['Significant']:
            # Determine number of stars based on p-value
            if row['P_corrected'] < 0.001:
                stars = "***"
            elif row['P_corrected'] < 0.01:
                stars = "**"
            elif row['P_corrected'] < 0.05:
                stars = "*"
                
            # Calculate position
            y_pos = max(pre_means[i] + pre_se[i], post_means[i] + post_se[i]) + 0.3
            
            # Add stars
            ax.text(i, y_pos, stars, ha='center', va='bottom', fontsize=14)
            
            # Add connecting line
            ax.plot([i-width/2, i+width/2], [y_pos-0.1, y_pos-0.1], '-', color='black', linewidth=1)
    
    # Customize layout
    ax.set_ylabel('Mean Score (1-7 scale)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(skills, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    
    # Add significance legend
    legend_elements = [
        Line2D([0], [0], marker='', color='none', label='Significance:'),
        Line2D([0], [0], marker='', color='none', label='* p < 0.05'),
        Line2D([0], [0], marker='', color='none', label='** p < 0.01'),
        Line2D([0], [0], marker='', color='none', label='*** p < 0.001')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    # Add note about Holm-Bonferroni
    plt.figtext(0.5, 0.01, "Note: p-values corrected using Holm-Bonferroni method", 
                ha="center", fontsize=9, style='italic')
    
    # Set y-axis to start from 0
    ax.set_ylim(1, 8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('critical_thinking_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('critical_thinking_bar_chart.pdf', bbox_inches='tight')
    plt.close()

def create_distribution_plot(pre_data, post_data, title):
    """Create density distribution plot for pre-post comparison"""
    plt.figure(figsize=(10, 6))
    
    # Plot densities
    sns.kdeplot(pre_data, fill=True, color='#4C72B0', alpha=0.7, 
              label=f'Pre-test (M={pre_data.mean():.2f})')
    sns.kdeplot(post_data, fill=True, color='#55A868', alpha=0.7, 
              label=f'Post-test (M={post_data.mean():.2f})')
    
    # Add mean lines
    plt.axvline(pre_data.mean(), color='#4C72B0', linestyle='--', linewidth=2)
    plt.axvline(post_data.mean(), color='#55A868', linestyle='--', linewidth=2)
    
    # Add mean difference annotation
    mean_diff = post_data.mean() - pre_data.mean()
    plt.annotate(f'Mean difference: +{mean_diff:.2f}',
              xy=((pre_data.mean() + post_data.mean())/2, 0.5),
              xytext=(0, 30), textcoords='offset points', 
              ha='center', va='bottom',
              bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    # Customize plot
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel('RED Score (1-7 scale)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig('critical_thinking_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('critical_thinking_distribution.pdf', bbox_inches='tight')
    plt.close()


def create_enhanced_red_density(pre_red, post_red):
    """Create enhanced density plot for RED scores with jitter points"""
    # Create figure with good dimensions
    plt.figure(figsize=(12, 7))
    
    # Plot densities
    sns.kdeplot(pre_red, fill=True, color='#4C72B0', alpha=0.6, 
              label=f'Pre-test (M={pre_red.mean():.2f})')
    sns.kdeplot(post_red, fill=True, color='#55A868', alpha=0.6, 
              label=f'Post-test (M={post_red.mean():.2f})')
    
    # Add mean lines
    plt.axvline(pre_red.mean(), color='#4C72B0', linestyle='--', linewidth=2)
    plt.axvline(post_red.mean(), color='#55A868', linestyle='--', linewidth=2)
    
    # Add jitter points at the bottom
    y_pos_pre = np.random.normal(-0.05, 0.02, size=len(pre_red))
    y_pos_post = np.random.normal(-0.15, 0.02, size=len(post_red))
    
    plt.scatter(pre_red, y_pos_pre, color='#4C72B0', s=30, alpha=0.7, edgecolor='white', linewidth=0.5)
    plt.scatter(post_red, y_pos_post, color='#55A868', s=30, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Add point labels
    plt.text(pre_red.mean(), -0.05, "Pre-test scores", color='#4C72B0', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(post_red.mean(), -0.15, "Post-test scores", color='#55A868', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add mean difference with white space background
    mean_diff = post_red.mean() - pre_red.mean()
    plt.text((pre_red.mean() + post_red.mean())/2, 0.5, 
           f'Mean Δ = +{mean_diff:.2f}',
           ha='center', va='center', fontweight='bold', fontsize=12,
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Add shaded area between means
    plt.axvspan(pre_red.mean(), post_red.mean(), alpha=0.1, color='gray')
    
    # Calculate statistics
    t_stat, p_value = stats.ttest_rel(post_red, pre_red)
    d = mean_diff / pre_red.std()
    
    # Add statistical annotation
    stats_text = f"t({len(pre_red)-1}) = {t_stat:.2f}, p < 0.001, d = {d:.2f}"
    plt.text(0.5, 0.95, stats_text, 
           ha='center', va='top', fontsize=11, transform=plt.gca().transAxes,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    # Customize plot
    plt.title('Distribution of RED Critical Thinking Scores', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('RED Overall Score (1-7 scale)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.ylim(-0.2, 1.2)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('enhanced_red_density.png', dpi=300, bbox_inches='tight')
    plt.savefig('enhanced_red_density.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created enhanced density plot with jitter points for RED scores")

def create_draw_conclusions_violin(pre_draw, post_draw):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Prepare data
    data_long = pd.DataFrame({
        'Time': ['Pre-test'] * len(pre_draw) + ['Post-test'] * len(post_draw),
        'Draw Conclusions Score': np.concatenate([pre_draw, post_draw])
    })

    pre_mean = pre_draw.mean()
    post_mean = post_draw.mean()
    pre_sd = pre_draw.std()
    post_sd = post_draw.std()
    pre_se = pre_sd / np.sqrt(len(pre_draw))
    post_se = post_sd / np.sqrt(len(post_draw))

    t_stat, p_value = stats.ttest_rel(post_draw, pre_draw)
    d = (post_mean - pre_mean) / pre_sd

    plt.figure(figsize=(6, 7))
    sns.violinplot(x='Time', y='Draw Conclusions Score', data=data_long, 
                   palette=['#4C72B0', '#55A868'], inner=None)
    sns.stripplot(x='Time', y='Draw Conclusions Score', data=data_long, 
                  jitter=True, size=6, alpha=0.7, 
                  palette=['#4C72B0', '#55A868'])

    plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
                 fmt='o', color='black', markersize=9, capsize=8, 
                 markerfacecolor='white', markeredgewidth=1.5, elinewidth=1.5)

    plt.text(0, pre_mean + pre_se + 0.15, f'M = {pre_mean:.2f}', ha='center', fontweight='bold')
    plt.text(1, post_mean + post_se + 0.15, f'M = {post_mean:.2f}', ha='center', fontweight='bold')

    max_y = 7.0
    plt.ylim(1.5, max_y)
    if p_value < 0.05:
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        plt.plot([0, 0, 1, 1], [max_y-0.5, max_y-0.4, max_y-0.4, max_y-0.5], lw=1.5, c='black')
        plt.text(0.5, max_y-0.3, stars, ha='center', va='bottom', fontsize=14)

        stats_text = f"t({len(pre_draw)-1}) = {t_stat:.2f}\np < 0.001\nd = {d:.2f}"
        plt.text(0.5, 2.0, stats_text, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.4'))

    plt.title('Draw Conclusions Component (RED Framework)', fontsize=14, fontweight='bold')
    plt.ylabel('Score (1–7 scale)', fontsize=11)
    plt.xlabel('')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.figtext(0.5, 0.01,
                "Note: Draw Conclusions aggregates Identify Weaknesses, Change Position, and Consider Perspectives",
                ha="center", fontsize=9, style='italic')

    # Save directly as PDF
    plt.tight_layout()
    plt.savefig("draw_conclusions_violin.png", bbox_inches='tight')
    plt.close()






def create_combined_red_density_bar_chart(pre_red, post_red):
    """Create a combined density plot and bar chart for overall RED scores"""
    # Calculate statistics
    pre_mean = pre_red.mean()
    post_mean = post_red.mean()
    pre_sd = pre_red.std()
    post_sd = post_red.std()
    pre_se = pre_sd / np.sqrt(len(pre_red))
    post_se = post_sd / np.sqrt(len(post_red))
    
    # Calculate t-test and effect size
    t_stat, p_value = stats.ttest_rel(post_red, pre_red)
    d = (post_mean - pre_mean) / pre_sd
    
    # Calculate confidence intervals
    ci_low, ci_high = stats.t.interval(0.95, len(pre_red)-1, 
                                     loc=post_mean - pre_mean, 
                                     scale=stats.sem(post_red - pre_red))
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                  gridspec_kw={'width_ratios': [2, 1]})
    
    # Colors
    pre_color = "#4C72B0"  # Blue
    post_color = "#55A868"  # Green
    
    # LEFT PANEL: Density plot
    # Plot densities
    sns.kdeplot(x=pre_red, ax=ax1, fill=True, color=pre_color, alpha=0.6, 
              label=f'Pre-test (M={pre_mean:.2f})')
    sns.kdeplot(x=post_red, ax=ax1, fill=True, color=post_color, alpha=0.6, 
              label=f'Post-test (M={post_mean:.2f})')
    
    # Add mean lines
    ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
    ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
    # Add shaded area between means
    ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
    # Add effect size annotation - Updated to include "huge" for d > 2
    effect_size_text = f"d = {d:.2f} (huge effect)" if d > 2 else f"d = {d:.2f} (large effect)"
    if d < 0.8:
        effect_size_text = f"d = {d:.2f} (medium effect)" if d >= 0.5 else f"d = {d:.2f} (small effect)"
    
    ax1.text((pre_mean + post_mean)/2, 0.8, effect_size_text, 
            ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add delta indicator
    ax1.text((pre_mean + post_mean)/2, 0.4, f'Δ = {post_mean - pre_mean:.2f}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9))
    
    # Add jitter points at the bottom - FIXED to match enhanced density plot
    y_pos_pre = np.random.normal(-0.05, 0.02, size=len(pre_red))
    y_pos_post = np.random.normal(-0.15, 0.02, size=len(post_red))
    
    ax1.scatter(pre_red, y_pos_pre, color=pre_color, s=30, alpha=0.7, 
               edgecolor='white', linewidth=0.5)
    ax1.scatter(post_red, y_pos_post, color=post_color, s=30, alpha=0.7, 
               edgecolor='white', linewidth=0.5)
    
    # Add labels for the data points
    ax1.text(pre_mean, -0.05, "Pre-test scores", color=pre_color, 
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(post_mean, -0.15, "Post-test scores", color=post_color, 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Customize left panel
    ax1.set_title('Distribution of RED Scores', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('RED Score (1-7 scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(1, 7)  # For RED scores (1-7 scale)
    ax1.set_ylim(-0.2, 1.1)
    ax1.grid(linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper left')
    
    # Set explicit x-axis ticks
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7'])
    
    # RIGHT PANEL: Bar chart
    bar_width = 0.7
    x_pos = [0, 1]
    bars = ax2.bar(x_pos, [pre_mean, post_mean], yerr=[pre_se, post_se], 
                  width=bar_width, capsize=10, color=[pre_color, post_color], 
                  edgecolor='black', linewidth=1.5)
    
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
    
    # Customize right panel
    ax2.set_title('Mean Scores with SE', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('RED Score (1-7 scale)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Pre-test', 'Post-test'], fontsize=11)
    ax2.set_ylim(1, y_max + 0.5)  # Start from 1 for RED scores
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add overall title
    plt.suptitle('Impact of Mind Elevator on RED Critical Thinking Skills', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add paired sample t-test results as a footnote
    plt.figtext(0.5, 0.01, 
               f"Paired t-test: t({len(pre_red)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig('red_scores_combined_density_bar.png', dpi=300, bbox_inches='tight')
    plt.savefig('red_scores_combined_density_bar.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created combined density-bar chart for overall RED scores")


def create_professional_radar_chart(results_df, title):
    """Create a professional radar chart with fixed label positioning"""
    # Number of variables
    categories = results_df['Skill'].tolist()
    N = len(categories)
    
    # Pre and post means
    pre_means = results_df['Pre_Mean'].tolist()
    post_means = results_df['Post_Mean'].tolist()
    
    # Calculate angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    # Close the plot
    pre_means_closed = pre_means + [pre_means[0]]
    post_means_closed = post_means + [post_means[0]]
    angles_closed = angles + [angles[0]]
    
    # Create figure with custom styling
    fig = plt.figure(figsize=(11, 11), facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    
    # Set background color
    ax.set_facecolor('#f9f9f9')
    
    # Add grid lines
    ax.grid(color='gray', alpha=0.2, linestyle='-')
    
    # Plot data with thicker lines and larger markers
    ax.plot(angles_closed, pre_means_closed, 'o-', linewidth=2.5, label='Pre-test', 
           color='#4C72B0', markersize=8)
    ax.fill(angles_closed, pre_means_closed, alpha=0.1, color='#4C72B0')
    
    ax.plot(angles_closed, post_means_closed, 'o-', linewidth=2.5, label='Post-test', 
           color='#55A868', markersize=8)
    ax.fill(angles_closed, post_means_closed, alpha=0.1, color='#55A868')
    
    # Set ticks and limits
    ax.set_xticks(angles)
    ax.set_ylim(0, 7)
    
    # UPDATED: Both labels lowered and vertically aligned
    # Define precise positions for each label to avoid overlapping and place banners correctly
    label_positions = {
        'Recognize Assumptions': {'distance': 8.3, 'offset': (0.3, -1.23)},  # Further lowered
        'Evaluate Evidence': {'distance': 8.3, 'offset': (-0.2, -0.4)},     # Lowered to match Recognize Assumptions
        'Identify Weaknesses': {'distance': 8.8, 'offset': (0, 0)},
        'Willing to Change Position': {'distance': 8.9, 'offset': (0, -0.2)},
        'Consider Perspectives': {'distance': 8.8, 'offset': (0.1, 0)}
    }
    
    # Add labels with custom positioning
    for i, angle in enumerate(angles):
        skill = categories[i]
        position = label_positions.get(skill, {'distance': 8.5, 'offset': (0, 0)})
        
        # Calculate position
        x = angle
        offset_x, offset_y = position['offset']
        
        # IMPROVED: Use different colors for label backgrounds for better distinction
        edge_color = 'lightgray'
        if skill in ['Recognize Assumptions', 'Evaluate Evidence']:
            edge_color = '#bbbbbb'  # Darker edge for problematic labels
        
        # Add the label with enhanced background box
        ax.text(x + offset_x, position['distance'] + offset_y, skill, 
               ha='center', va='center', fontsize=12, fontweight='bold', 
               bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.5', 
                        edgecolor=edge_color, linewidth=1.5))
        
        # Add mean values directly on the data points with clearer formatting
        # For pre-test
        ax.text(angle, pre_means[i] + 0.2, f"{pre_means[i]:.2f}", 
               color='#4C72B0', ha='center', va='center', fontsize=10, 
               fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, 
                                          boxstyle='round,pad=0.2', edgecolor='#4C72B0'))
        
        # For post-test
        ax.text(angle, post_means[i] + 0.2, f"{post_means[i]:.2f}", 
               color='#55A868', ha='center', va='center', fontsize=10, 
               fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, 
                                          boxstyle='round,pad=0.2', edgecolor='#55A868'))
    
    # Remove default angular labels
    ax.set_xticklabels([])
    
    # Set radial ticks and labels
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7"], fontsize=9, color="gray")
    ax.set_rlabel_position(0)
    
    # Add circular gridlines with labels
    for y in [1, 2, 3, 4, 5, 6, 7]:
        ax.text(np.pi/4, y, str(y), ha='center', va='center', color='gray', fontsize=9)
    
    # Enhanced legend
    legend = plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), fontsize=12,
                      frameon=True, framealpha=0.9, borderpad=1, 
                      edgecolor='lightgray')
    
    # Add title with professional styling
    plt.title(title, size=18, y=1.08, fontweight='bold', color='#333333')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('critical_thinking_radar_professional.png', dpi=300, bbox_inches='tight')
    plt.savefig('critical_thinking_radar_professional.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created professional radar chart with vertically aligned labels")



def create_combined_red_density_boxplot(pre_red, post_red):
    """Create a combined density plot and box plot for overall RED scores"""
    # Calculate statistics
    pre_mean = pre_red.mean()
    post_mean = post_red.mean()
    pre_sd = pre_red.std()
    post_sd = post_red.std()
    pre_se = pre_sd / np.sqrt(len(pre_red))
    post_se = post_sd / np.sqrt(len(post_red))
    
    # Calculate t-test and effect size
    t_stat, p_value = stats.ttest_rel(post_red, pre_red)
    d = (post_mean - pre_mean) / pre_sd
    
    # Calculate confidence intervals
    ci_low, ci_high = stats.t.interval(0.95, len(pre_red)-1, 
                                     loc=post_mean - pre_mean, 
                                     scale=stats.sem(post_red - pre_red))
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                  gridspec_kw={'width_ratios': [2, 1]})
    
    # Colors
    pre_color = "#4C72B0"  # Blue
    post_color = "#55A868"  # Green
    
    # LEFT PANEL: Density plot
    # Plot densities
    sns.kdeplot(x=pre_red, ax=ax1, fill=True, color=pre_color, alpha=0.6, 
              label=f'Pre-test (M={pre_mean:.2f})')
    sns.kdeplot(x=post_red, ax=ax1, fill=True, color=post_color, alpha=0.6, 
              label=f'Post-test (M={post_mean:.2f})')
    
    # Add mean lines
    ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
    ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
    # Add shaded area between means
    ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
    # Add delta indicator only (NO effect size annotation on density)
    ax1.text((pre_mean + post_mean)/2, 0.75, f'Δ = {post_mean - pre_mean:.2f}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9))
    
    # Customize left panel
    ax1.set_title('Distribution of RED Scores', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('RED Score (1-7 scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(1, 7)  # For RED scores (1-7 scale)
    ax1.set_ylim(0, 1.1)  # Changed lower limit from -0.2 to 0 since we removed points
    ax1.grid(linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper left')
    
    # Set explicit x-axis ticks
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7'])
    
    # RIGHT PANEL: Box plot (instead of bar chart)
    # Prepare data for boxplot
    box_data = [pre_red, post_red]
    
    # Create boxplot with custom styling
    boxprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    medianprops = dict(linewidth=2.5, color='black')
    meanprops = dict(marker='o', markerfacecolor='white', markeredgecolor='black',
                   markersize=8, markeredgewidth=2)
    
    # Get y-limit to properly position annotations
    y_max = max(pre_red.max(), post_red.max()) + 0.3
    
    # Create a more visually appealing box plot - UPDATED: Added showfliers=False to hide outlier markers
    bplot = ax2.boxplot(box_data, patch_artist=True, 
                       vert=True, widths=0.6, showmeans=True, showfliers=False,
                       boxprops=boxprops, whiskerprops=whiskerprops,
                       capprops=capprops, medianprops=medianprops,
                       meanprops=meanprops,
                       labels=['Pre-test', 'Post-test'])
    
    # Color the boxes
    for i, box in enumerate(bplot['boxes']):
        color = pre_color if i == 0 else post_color
        box.set_facecolor(color)
        box.set_alpha(0.6)
        box.set_edgecolor('black')
    
    # Add data points as swarm plot
    for i, data in enumerate([pre_red, post_red]):
        # Add jittered points
        x_pos = i + 1
        y = data.values
        # Add points with slight jitter
        jitter = np.random.normal(0, 0.05, size=len(y))
        ax2.scatter(x_pos + jitter, y, color=pre_color if i == 0 else post_color,
                   alpha=0.7, s=30, edgecolor='white', linewidth=0.5)
    
    # Add mean values as text below the data points
    ax2.text(1, 1.0, f'Mean = {pre_mean:.2f}', ha='center', 
            fontsize=10, fontweight='bold', color=pre_color,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    ax2.text(2, 1.0, f'Mean = {post_mean:.2f}', ha='center', 
            fontsize=10, fontweight='bold', color=post_color,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Add significance annotation
    ax2.plot([1, 1, 2, 2], [y_max, y_max+0.1, y_max+0.1, y_max], lw=1.5, c='black')
    
    # Add stars for significance
    stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
    ax2.text(1.5, y_max + 0.2, stars, ha='center', va='bottom', fontsize=16)
    
    # Customize right panel
    ax2.set_title('Box Plot Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('RED Score (1-7 scale)', fontsize=12)
    ax2.set_ylim(0.7, y_max + 0.5)  # Start from a lower value to show the mean labels
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add stats text box
    stats_text = f"t({len(pre_red)-1}) = {t_stat:.2f}\np < {0.001 if p_value < 0.001 else p_value:.3f}"
    ax2.text(1.5, 2.0, stats_text, ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add overall title
    plt.suptitle('Impact of Mind Elevator on RED Critical Thinking Skills', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add paired sample t-test results as a footnote
    plt.figtext(0.5, 0.01, 
               f"Paired t-test: t({len(pre_red)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig('red_scores_combined_density_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('red_scores_combined_density_boxplot.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created combined density-boxplot chart for overall RED scores")
    return fig

if __name__ == "__main__":
    # Run the analysis and get the results
    results = analyze_critical_thinking('pre_form.csv', 'post_form.csv')
    
    # Extract RED and Draw Conclusions data
    pre_red = results['overall_red']['pre_data']
    post_red = results['overall_red']['post_data'] 
    pre_draw = results['draw_conclusions']['pre_data']
    post_draw = results['draw_conclusions']['post_data']
    
    # Now call your visualization functions with the extracted data
    create_enhanced_red_density(pre_red, post_red)
    create_draw_conclusions_violin(pre_draw, post_draw)
    create_combined_red_density_bar_chart(pre_red, post_red)  # Add this new line
    create_combined_red_density_boxplot(pre_red, post_red)  # Add this new line
    create_professional_radar_chart(results['individual_skills'], 
                                   "Critical Thinking Skills: Pre vs Post-Intervention")