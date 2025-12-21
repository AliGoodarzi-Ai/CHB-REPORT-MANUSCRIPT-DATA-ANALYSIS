import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def analyze_argumentation_efficacy(pre_csv, post_csv):
    """
    Comprehensive analysis of argumentation skills and self-efficacy (RQ3)
    """
    # Load data
    pre_df = pd.read_csv(pre_csv)
    post_df = pd.read_csv(post_csv)
    
    print("Analyzing argumentation skills and self-efficacy (RQ3)...\n")
    
    # Define the specific columns for Toulmin model components
    toulmin_components = {
        'Claim': 'Claim',
        'Grounds': 'Grounds',
        'Warrant': 'Warrant',
        'Backing': 'Backing',
        'Qualifier': 'Qualifier',
        'Rebuttals': 'Rebuttals'
    }
    
    # Define self-efficacy column
    self_efficacy_col = 'self_ability_argument'
    
    # Check columns exist in datasets
    for component_name, col in toulmin_components.items():
        if col not in pre_df.columns or col not in post_df.columns:
            print(f"ERROR: '{col}' column not found in one or both datasets")
            return
    
    if self_efficacy_col not in pre_df.columns or self_efficacy_col not in post_df.columns:
        print(f"ERROR: '{self_efficacy_col}' column not found in one or both datasets")
        return
    
    # Create dataframe to store all analysis results
    results_df = pd.DataFrame(columns=[
        'Component', 'Pre_Mean', 'Pre_SD', 'Post_Mean', 'Post_SD', 
        'Mean_Diff', 'Percent_Change', 'T_stat', 'P_value', 
        'Cohens_d', 'CI_Low', 'CI_High', 'Improved_Pct'
    ])
    
    # Store p-values for Holm-Bonferroni correction
    p_values = []
    component_names = []
    
    # 1. Process each Toulmin component
    for component_name, col in toulmin_components.items():
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
        component_names.append(component_name)
        
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
            'Component': [component_name],
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
    
    # 2. Process self-efficacy
    pre_efficacy = pre_df[self_efficacy_col].astype(float)
    post_efficacy = post_df[self_efficacy_col].astype(float)
    
    # Basic statistics for self-efficacy
    pre_eff_mean = pre_efficacy.mean()
    pre_eff_sd = pre_efficacy.std()
    post_eff_mean = post_efficacy.mean()
    post_eff_sd = post_efficacy.std()
    eff_mean_diff = post_eff_mean - pre_eff_mean
    eff_percent_change = (eff_mean_diff / pre_eff_mean) * 100
    
    # Paired t-test for self-efficacy
    t_eff, p_eff = stats.ttest_rel(post_efficacy, pre_efficacy)
    
    # Add self-efficacy p-value to list for correction
    p_values.append(p_eff)
    component_names.append('Self-Efficacy')
    
    # Effect size for self-efficacy
    d_eff = eff_mean_diff / pre_eff_sd
    
    # Confidence interval for self-efficacy
    ci_low_eff, ci_high_eff = stats.t.interval(
        0.95, len(pre_efficacy)-1, 
        loc=eff_mean_diff, 
        scale=stats.sem(post_efficacy - pre_efficacy)
    )
    
    # Percentage improved for self-efficacy
    improved_eff = np.sum(post_efficacy > pre_efficacy)
    improved_eff_pct = (improved_eff / len(pre_efficacy)) * 100
    
    # Add self-efficacy to results
    results_df = pd.concat([results_df, pd.DataFrame({
        'Component': ['Self-Efficacy'],
        'Pre_Mean': [pre_eff_mean],
        'Pre_SD': [pre_eff_sd],
        'Post_Mean': [post_eff_mean],
        'Post_SD': [post_eff_sd],
        'Mean_Diff': [eff_mean_diff],
        'Percent_Change': [eff_percent_change],
        'T_stat': [t_eff],
        'P_value': [p_eff],
        'Cohens_d': [d_eff],
        'CI_Low': [ci_low_eff],
        'CI_High': [ci_high_eff],
        'Improved_Pct': [improved_eff_pct]
    })], ignore_index=True)
    
    # 3. Calculate composite Toulmin score
    pre_toulmin = pre_df[list(toulmin_components.values())].mean(axis=1)
    post_toulmin = post_df[list(toulmin_components.values())].mean(axis=1)
    
    # Statistics for composite Toulmin
    pre_toulmin_mean = pre_toulmin.mean()
    pre_toulmin_sd = pre_toulmin.std()
    post_toulmin_mean = post_toulmin.mean()
    post_toulmin_sd = post_toulmin.std()
    toulmin_mean_diff = post_toulmin_mean - pre_toulmin_mean
    
    # T-test for composite Toulmin
    t_toulmin, p_toulmin = stats.ttest_rel(post_toulmin, pre_toulmin)
    d_toulmin = toulmin_mean_diff / pre_toulmin_sd
    
    # 4. Apply Holm-Bonferroni correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
    results_df['P_corrected'] = p_corrected
    results_df['Significant'] = reject
    
    # Print detailed results
    print("\n=== ARGUMENTATION SKILLS & SELF-EFFICACY ANALYSIS (RQ3) ===")
    print("\nIndividual Components Analysis:")
    print("-" * 100)
    print(f"{'Component':<20} {'Pre':<10} {'Post':<10} {'Change':<10} {'t-stat':<10} {'p-value':<10} {'p-corr':<10} {'Sig?':<6} {'d':<8}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        sig_symbol = "***" if row['P_corrected'] < 0.001 else "**" if row['P_corrected'] < 0.01 else "*" if row['P_corrected'] < 0.05 else ""
        print(f"{row['Component']:<20} {row['Pre_Mean']:.2f}±{row['Pre_SD']:.2f} {row['Post_Mean']:.2f}±{row['Post_SD']:.2f} {row['Mean_Diff']:+.2f} {row['T_stat']:.2f} {row['P_value']:.4f} {row['P_corrected']:.4f} {sig_symbol:<6} {row['Cohens_d']:.2f}")
    
    print("-" * 100)
    print("\nComposite Scores Analysis:")
    print(f"Overall Toulmin score: {pre_toulmin_mean:.2f}±{pre_toulmin_sd:.2f} → {post_toulmin_mean:.2f}±{post_toulmin_sd:.2f}, t={t_toulmin:.2f}, p={p_toulmin:.4f}, d={d_toulmin:.2f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Enhanced visualizations for argumentation analysis
    create_individual_violin_charts(pre_df, post_df, toulmin_components, results_df)
    create_self_efficacy_violin(pre_efficacy, post_efficacy, results_df)
    create_combined_toulmin_violin(pre_df, post_df, toulmin_components, results_df)
    create_toulmin_radar_chart(results_df.iloc[:6], "Toulmin Model Components: Pre vs Post")
    create_bar_chart(results_df, "Changes in Argumentation Skills & Self-Efficacy")
    visualize_holm_bonferroni(results_df)
    
    # Save results to CSV
    results_df.to_csv('argumentation_analysis_results.csv', index=False)
    print("\nResults saved to 'argumentation_analysis_results.csv'")
    
    # Return results for further analysis if needed
    return {
        'individual_components': results_df,
        'composite_toulmin': {
            'pre_mean': pre_toulmin_mean,
            'pre_sd': pre_toulmin_sd,
            'post_mean': post_toulmin_mean,
            'post_sd': post_toulmin_sd,
            't_stat': t_toulmin,
            'p_value': p_toulmin,
            'cohens_d': d_toulmin
        }
    }

def create_individual_violin_charts(pre_df, post_df, components, results_df):
    """Create separate IEEE-friendly violin charts for each argumentation component"""
    for component_name, col in components.items():
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
        
        # Get corrected p-value from results_df
        p_corrected = results_df.loc[results_df['Component'] == component_name, 'P_corrected'].values[0]
        
        # Create figure
        plt.figure(figsize=(7, 8))
        
        # Create violin plot (with clear separation)
        ax = sns.violinplot(x='Time', y='Score', data=data_long, 
                         inner=None, 
                         palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'})
        
        # Add individual data points
        sns.stripplot(x='Time', y='Score', data=data_long, 
                  jitter=True, size=6, alpha=0.7,
                  palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'})
        
        # Add mean points with error bars
        plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
                   fmt='o', color='black', markersize=10, capsize=10, 
                   markerfacecolor='white', markeredgewidth=2, elinewidth=2)
        
        # Add mean values as text
        plt.text(0, pre_mean + pre_se + 0.2, f'M = {pre_mean:.2f}', ha='center', fontweight='bold')
        plt.text(1, post_mean + post_se + 0.2, f'M = {post_mean:.2f}', ha='center', fontweight='bold')
        
        # Add significance indicators with Holm-Bonferroni note
        max_y = max(pre_data.max(), post_data.max()) + 0.5
        if p_corrected < 0.05:
            stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*"
            plt.plot([0, 0, 1, 1], [max_y, max_y+0.1, max_y+0.1, max_y], lw=1.5, c='black')
            plt.text(0.5, max_y+0.2, stars, ha='center', va='bottom', fontsize=16)
            
            # Add statistics text in box
            stats_text = f"t({len(pre_data)-1}) = {t_stat:.2f}\np = {p_corrected:.3f}, d = {d:.2f}"
            plt.text(0.5, 2.0, stats_text, ha='center', va='center',
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Set title and labels
        plt.title(f'Toulmin Component: {component_name}', fontsize=15, pad=20)
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
        filename = f"violin_toulmin_{component_name.lower()}"
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{filename}.pdf', bbox_inches='tight')
        plt.close()
        
    print("Created IEEE-friendly violin charts for each Toulmin component")

def create_self_efficacy_violin(pre_efficacy, post_efficacy, results_df):
    """Create violin chart for self-efficacy in argumentation"""
    # Prepare data for plotting
    data_long = pd.DataFrame({
        'Time': ['Pre-test'] * len(pre_efficacy) + ['Post-test'] * len(post_efficacy),
        'Self-Efficacy Score': np.concatenate([pre_efficacy, post_efficacy])
    })
    
    # Calculate statistics
    pre_mean = pre_efficacy.mean()
    pre_sd = pre_efficacy.std()
    post_mean = post_efficacy.mean()
    post_sd = post_efficacy.std()
    pre_se = pre_sd / np.sqrt(len(pre_efficacy))
    post_se = post_sd / np.sqrt(len(post_efficacy))
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(post_efficacy, pre_efficacy)
    d = (post_mean - pre_mean) / pre_sd
    
    # Get corrected p-value from results_df
    p_corrected = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'P_corrected'].values[0]
    
    # Create figure
    plt.figure(figsize=(8, 9))
    
    # Create violin plot
    ax = sns.violinplot(x='Time', y='Self-Efficacy Score', data=data_long, 
                      inner=None, 
                      palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'})
    
    # Add scatter points with jitter
    sns.stripplot(x='Time', y='Self-Efficacy Score', data=data_long, 
                jitter=True, size=7, alpha=0.7,
                palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'})
    
    # Add mean points with error bars
    plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
                fmt='o', color='black', markersize=10, capsize=10, 
                markerfacecolor='white', markeredgewidth=2, elinewidth=2)
    
    # Add mean values as text
    plt.text(0, pre_mean + pre_se + 0.2, f'M = {pre_mean:.2f}', ha='center', fontweight='bold')
    plt.text(1, post_mean + post_se + 0.2, f'M = {post_mean:.2f}', ha='center', fontweight='bold')
    
    # Add significance annotation
    max_y = max(pre_efficacy.max(), post_efficacy.max()) + 0.5
    if p_corrected < 0.05:
        stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*"
        plt.plot([0, 0, 1, 1], [max_y, max_y+0.1, max_y+0.1, max_y], lw=1.5, c='black')
        plt.text(0.5, max_y+0.2, stars, ha='center', va='bottom', fontsize=16)
        
        # Add statistics box
        stats_text = f"t({len(pre_efficacy)-1}) = {t_stat:.2f}\np = {p_corrected:.3f}\nd = {d:.2f}"
        plt.text(0.5, 2.0, stats_text, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set title and labels
    plt.title('Self-Efficacy in Argumentation', fontsize=15, pad=20)
    plt.ylabel('Score (1-7 scale)', fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.ylim(1.0, max_y+1.0)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add Holm-Bonferroni note
    plt.figtext(0.5, 0.01, "Note: p-value corrected using Holm-Bonferroni method", 
              ha="center", fontsize=9, style='italic')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('self_efficacy_violin.png', dpi=300, bbox_inches='tight')
    plt.savefig('self_efficacy_violin.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created violin plot for self-efficacy in argumentation")

def create_combined_toulmin_violin(pre_df, post_df, components, results_df):
    """Create a single violin plot with all Toulmin components"""
    # Prepare data in long format
    data_list = []
    
    for component_name, col in components.items():
        pre_data = pre_df[col].astype(float)
        post_data = post_df[col].astype(float)
        
        for i, value in enumerate(pre_data):
            data_list.append({
                'Component': component_name,
                'Time': 'Pre-test',
                'Score': value
            })
        
        for i, value in enumerate(post_data):
            data_list.append({
                'Component': component_name,
                'Time': 'Post-test',
                'Score': value
            })
    
    # Convert to DataFrame
    data_long = pd.DataFrame(data_list)
    
    # Create plot
    plt.figure(figsize=(16, 8))
    
    # Use dodge=True for clearer separation between pre/post
    ax = sns.violinplot(x='Component', y='Score', hue='Time', 
                     data=data_long, dodge=True,
                     palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'})
    
    # Add individual data points
    sns.stripplot(x='Component', y='Score', hue='Time', 
               data=data_long, dodge=True, alpha=0.3, size=4,
               palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'})
    
    # Add significance indicators
    for i, component_name in enumerate(components.keys()):
        # Get p-value from results_df for this component
        p_value = results_df.loc[results_df['Component'] == component_name, 'P_corrected'].values[0]
        
        # Add stars based on p-value
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        if stars:
            plt.text(i, 7.5, stars, ha='center', va='bottom', fontsize=16)
    
    # Customize plot
    plt.title('Toulmin Model Components: Pre-test vs Post-test', fontsize=16, pad=20)
    plt.ylabel('Score (1-7 scale)', fontsize=14)
    plt.xlabel('', fontsize=14)
    plt.ylim(1, 8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Fix legend (avoid duplicate entries)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right', fontsize=12)
    
    # Add note about Holm-Bonferroni
    plt.figtext(0.5, 0.01, "Note: * p < 0.05, ** p < 0.01, *** p < 0.001 (Holm-Bonferroni corrected)", 
              ha="center", fontsize=10, style='italic')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('toulmin_combined_violin.png', dpi=300, bbox_inches='tight')
    plt.savefig('toulmin_combined_violin.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created combined violin plot for all Toulmin components")

def create_toulmin_radar_chart(results_df, title):
    """Create radar chart with non-overlapping text labels for Toulmin components"""
    # Number of variables
    categories = results_df['Component'].tolist()
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
    plt.savefig('toulmin_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('toulmin_radar_chart.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created improved radar chart for Toulmin components")

def create_bar_chart(results_df, title):
    """Create bar chart with error bars and significance indicators"""
    # Setup
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data
    components = results_df['Component'].tolist()
    pre_means = results_df['Pre_Mean'].tolist()
    post_means = results_df['Post_Mean'].tolist()
    pre_sds = results_df['Pre_SD'].tolist()
    post_sds = results_df['Post_SD'].tolist()
    
    # Calculate standard errors
    n = 16  # Number of participants
    pre_se = [sd / np.sqrt(n) for sd in pre_sds]
    post_se = [sd / np.sqrt(n) for sd in post_sds]
    
    # X positions
    x = np.arange(len(components))
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
    ax.set_xticklabels(components, rotation=15, ha='right', fontsize=10)
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
    plt.savefig('toulmin_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('toulmin_bar_chart.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created bar chart for all components")

def visualize_holm_bonferroni(results_df):
    """Create visualization and explanation of Holm-Bonferroni correction"""
    # Extract data
    components = results_df['Component'].tolist()
    p_values = results_df['P_value'].tolist()
    p_corrected = results_df['P_corrected'].tolist()
    is_significant = results_df['Significant'].tolist()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Sort by p-value for proper Holm-Bonferroni visualization
    sorted_indices = np.argsort(p_values)
    sorted_components = [components[i] for i in sorted_indices]
    sorted_p = [p_values[i] for i in sorted_indices]
    sorted_p_corr = [p_corrected[i] for i in sorted_indices]
    sorted_sig = [is_significant[i] for i in sorted_indices]
    
    # Create bar chart
    bar_width = 0.35
    x = np.arange(len(sorted_components))
    
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
        else:
            plt.text(i, 0.001, '✗', ha='center', va='center', fontsize=16, 
                   color='red', fontweight='bold')
    
    # Customize plot
    plt.yscale('log')  # Log scale for better visibility
    plt.ylabel('p-value (log scale)', fontsize=12)
    plt.xlabel('Components (ordered by p-value)', fontsize=12)
    plt.title('Holm-Bonferroni Correction for Multiple Comparisons', fontsize=14, pad=20, fontweight='bold')
    plt.xticks(x, sorted_components, rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, "Note: ✓ indicates significance maintained after correction (p < 0.05)", 
              ha="center", fontsize=10, style='italic')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('argumentation_holm_bonferroni.png', dpi=300, bbox_inches='tight')
    plt.savefig('argumentation_holm_bonferroni.pdf', bbox_inches='tight')
    plt.close()
    
    # Text explanation for Holm-Bonferroni (printed to console)
    print("\n=== HOLM-BONFERRONI CORRECTION EXPLANATION ===")
    print("The Holm-Bonferroni method controls the familywise error rate when conducting multiple hypothesis tests.")
    print("It works by ordering p-values from smallest to largest and applying sequential rejection:")
    print()
    print("Original significance threshold (α): 0.05")
    print(f"Number of tests performed: {len(sorted_components)}")
    print()
    print("Procedure:")
    for i, (component, p, p_corr, sig) in enumerate(zip(sorted_components, sorted_p, sorted_p_corr, sorted_sig)):
        alpha_adjusted = 0.05 / (len(sorted_components) - i)
        print(f"{i+1}. {component}: original p = {p:.6f}, adjusted α = {alpha_adjusted:.6f}, corrected p = {p_corr:.6f}")
        if sig:
            print("   ✓ Remains significant after correction")
        else:
            print("   ✗ No longer significant after correction")
            






def create_self_efficacy_bar_chart(pre_efficacy, post_efficacy, results_df):
    """Create a detailed bar chart with error bars for self-efficacy scores"""
    # Calculate statistics
    pre_mean = pre_efficacy.mean()
    post_mean = post_efficacy.mean()
    pre_sd = pre_efficacy.std()
    post_sd = post_efficacy.std()
    pre_se = pre_sd / np.sqrt(len(pre_efficacy))
    post_se = post_sd / np.sqrt(len(post_efficacy))
    
    # Get p-value from results_df
    p_value = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'P_value'].values[0]
    p_corrected = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'P_corrected'].values[0]
    d = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'Cohens_d'].values[0]
    
    # Create figure
    plt.figure(figsize=(8, 9))
    
    # Bar positions
    x_pos = [0, 1]
    width = 0.7
    
    # Plot bars
    bars = plt.bar(x_pos, [pre_mean, post_mean], yerr=[pre_se, post_se], 
                width=width, capsize=10, color=['#4C72B0', '#55A868'], 
                edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (pre_se if i==0 else post_se) + 0.15,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add significance indicator if significant
    if p_corrected < 0.05:
        stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*"
        y_pos = max(pre_mean + pre_se, post_mean + post_se) + 0.4
        plt.plot([0, 0, 1, 1], [y_pos-0.1, y_pos, y_pos, y_pos-0.1], 'k-', linewidth=1.5)
        plt.text(0.5, y_pos + 0.1, stars, ha='center', va='bottom', fontsize=16)
    
    # Add detailed statistics text box
    stats_text = (f"t-test: t({len(pre_efficacy)-1}) = {results_df.loc[results_df['Component'] == 'Self-Efficacy', 'T_stat'].values[0]:.2f}\n"
                 f"p = {p_corrected:.3f} (corrected)\n"
                 f"Cohen's d = {d:.2f}\n"
                 f"Pre: {pre_mean:.2f} ± {pre_sd:.2f}\n"
                 f"Post: {post_mean:.2f} ± {post_sd:.2f}\n"
                 f"Change: +{post_mean - pre_mean:.2f} ({((post_mean - pre_mean) / pre_mean * 100):.1f}%)")
    
    plt.text(0.5, 4.0, stats_text, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'),
           fontsize=11)
    
    # Customize plot
    plt.xticks(x_pos, ['Pre-test', 'Post-test'], fontsize=12)
    plt.ylabel('Self-Efficacy Score (1-7 scale)', fontsize=14)
    plt.title('Argumentation Self-Efficacy: Pre vs Post-Intervention', fontsize=16, pad=20)
    plt.ylim(1, 7.5)  # 1-7 scale with room for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add note about Holm-Bonferroni
    plt.figtext(0.5, 0.01, "Note: p-value corrected using Holm-Bonferroni method", 
              ha="center", fontsize=9, style='italic')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('self_efficacy_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('self_efficacy_bar_chart.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created detailed bar chart for self-efficacy with error bars")







def create_improved_toulmin_violin(pre_df, post_df, components, results_df):
    """Create an improved, clean violin plot with all Toulmin components"""
    # Set up the figure with enhanced styling
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()
    
    # Colors
    pre_color = '#4C72B0'  # Blue
    post_color = '#55A868'  # Green
    
    # Process each component
    for i, (component_name, col) in enumerate(components.items()):
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
        
        # Get corrected p-value
        p_corr = results_df.loc[results_df['Component'] == component_name, 'P_corrected'].values[0]
        d = results_df.loc[results_df['Component'] == component_name, 'Cohens_d'].values[0]
        
        # Create violin plot on the appropriate subplot
        ax = axes[i]
        
        # Plot violins with clear styling
        sns.violinplot(x='Time', y='Score', data=data_long, ax=ax, 
                      inner=None, 
                      hue='Time',
                      palette={'Pre-test': pre_color, 'Post-test': post_color},
                      legend=False)
        
        # Add individual data points
        sns.stripplot(x='Time', y='Score', data=data_long, ax=ax,
                    jitter=True, size=6, alpha=0.7,
                    hue='Time',
                    palette={'Pre-test': pre_color, 'Post-test': post_color},
                    legend=False)
        
        # Add mean markers
        ax.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se],
                  fmt='o', color='black', markersize=8, capsize=6, 
                  markerfacecolor='white', markeredgewidth=1.5)
        
        # Add mean values as text
        ax.text(0, pre_mean + pre_se + 0.15, f'{pre_mean:.2f}', ha='center', fontsize=10, fontweight='bold')
        ax.text(1, post_mean + post_se + 0.15, f'{post_mean:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        # Add significance stars if significant
        if p_corr < 0.05:
            stars = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*"
            y_pos = max(pre_mean + pre_se, post_mean + post_se) + 0.5
            ax.plot([0, 0, 1, 1], [y_pos, y_pos+0.1, y_pos+0.1, y_pos], color='black', lw=1.2)
            ax.text(0.5, y_pos+0.15, stars, ha='center', fontsize=14)
            
            # Add effect size
            effect_text = f"d = {d:.2f}"
            ax.text(0.5, y_pos+0.35, effect_text, ha='center', fontsize=10)
        
        # Set title and labels
        ax.set_title(component_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Score (1-7 scale)' if i % 3 == 0 else '')
        ax.set_xlabel('')
        ax.set_ylim(1, 7.5)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # REMOVED: Overall title
    # REMOVED: Footnote explaining significance
    
    # Save figure
    plt.tight_layout()
    # Fix to avoid invalid filename characters
    plt.savefig('toulmin_components_violin.png', dpi=300, bbox_inches='tight')
    plt.savefig('toulmin_components_violin.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created improved visualization of Toulmin components")



def create_toulmin_self_efficacy_correlation(pre_toulmin, post_toulmin, pre_efficacy, post_efficacy):
    """Create scatter plot showing correlation between Toulmin scores and self-efficacy"""
    # Calculate correlations
    pre_corr, pre_p = stats.pearsonr(pre_toulmin, pre_efficacy)
    post_corr, post_p = stats.pearsonr(post_toulmin, post_efficacy)
    
    # Calculate change scores and their correlation
    toulmin_change = post_toulmin - pre_toulmin
    efficacy_change = post_efficacy - pre_efficacy
    change_corr, change_p = stats.pearsonr(toulmin_change, efficacy_change)
    
    # Create a figure with single subplot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(toulmin_change, efficacy_change, color='#C44E52', s=80, alpha=0.7,
              edgecolor='white', linewidth=0.8)
    
    # Add best fit line
    m, b = np.polyfit(toulmin_change, efficacy_change, 1)
    x_line = np.array([min(toulmin_change), max(toulmin_change)])
    plt.plot(x_line, m * x_line + b, color='#C44E52', linestyle='--', linewidth=2)
    
    # Add correlation details - SIMPLIFIED P-VALUES
    sig_text = "" if change_p >= 0.05 else "*" if change_p < 0.05 else "**" if change_p < 0.01 else "***"
    plt.text(0.05, 0.95, f"r = {change_corr:.2f}{sig_text}\np = {change_p:.3f}", 
           transform=plt.gca().transAxes, ha='left', va='top', fontsize=12,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add quadrant labels
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.text(min(toulmin_change)/2, max(efficacy_change)/2, 
           "Lower Toulmin\nHigher Self-Efficacy", 
           ha='center', va='center', fontsize=10, alpha=0.7)
    
    plt.text(max(toulmin_change)/2, max(efficacy_change)/2, 
           "Higher Toulmin\nHigher Self-Efficacy", 
           ha='center', va='center', fontsize=10, alpha=0.7)
    
    plt.text(min(toulmin_change)/2, min(efficacy_change)/2, 
           "Lower Toulmin\nLower Self-Efficacy", 
           ha='center', va='center', fontsize=10, alpha=0.7)
    
    plt.text(max(toulmin_change)/2, min(efficacy_change)/2, 
           "Higher Toulmin\nLower Self-Efficacy", 
           ha='center', va='center', fontsize=10, alpha=0.7)
    
    # Customize plot
    plt.title('Relationship Between Changes in Argumentation Skills and Self-Efficacy', 
            fontsize=14, pad=20)
    plt.xlabel('Change in Overall Toulmin Score', fontsize=12)
    plt.ylabel('Change in Self-Efficacy Score', fontsize=12)
    plt.grid(linestyle='--', alpha=0.4)
    
    # Add statistics in a text box - SIMPLIFIED P-VALUES
    info_text = (f"Pre-test correlation: r = {pre_corr:.2f}, p = {pre_p:.3f}\n"
               f"Post-test correlation: r = {post_corr:.2f}, p = {post_p:.3f}\n"
               f"Change correlation: r = {change_corr:.2f}, p = {change_p:.3f}")
    
    # Reposition to top-right with simplified values
    plt.text(0.95, 0.95, info_text, ha='right', va='top', fontsize=10, 
           transform=plt.gca().transAxes,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    # Save figure
    plt.tight_layout()
    plt.savefig('toulmin_self_efficacy_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig('toulmin_self_efficacy_correlation.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created correlation analysis between changes in Toulmin scores and self-efficacy")







def create_combined_self_efficacy_visualization(pre_efficacy, post_efficacy, results_df):
    """Create a combined visualization with violin plot and bar chart for self-efficacy"""
    # Calculate statistics
    pre_mean = pre_efficacy.mean()
    post_mean = post_efficacy.mean()
    pre_sd = pre_efficacy.std()
    post_sd = post_efficacy.std()
    pre_se = pre_sd / np.sqrt(len(pre_efficacy))
    post_se = post_sd / np.sqrt(len(post_efficacy))
    
    # Get p-value from results_df
    p_value = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'P_value'].values[0]
    p_corrected = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'P_corrected'].values[0]
    d = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'Cohens_d'].values[0]
    
    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), 
                                  gridspec_kw={'width_ratios': [1, 1]})
    
    # Colors
    pre_color = '#4C72B0'  # Blue
    post_color = '#55A868'  # Green
    
    # LEFT PANEL: Violin plot
    # Prepare data for plotting
    data_long = pd.DataFrame({
        'Time': ['Pre-test'] * len(pre_efficacy) + ['Post-test'] * len(post_efficacy),
        'Self-Efficacy Score': np.concatenate([pre_efficacy, post_efficacy])
    })
    
    # Create violin plot
    sns.violinplot(x='Time', y='Self-Efficacy Score', data=data_long, ax=ax1,
                  inner=None, 
                  hue='Time',
                  palette={'Pre-test': pre_color, 'Post-test': post_color},
                  legend=False)
    
    # Add scatter points with jitter
    sns.stripplot(x='Time', y='Self-Efficacy Score', data=data_long, ax=ax1,
                jitter=True, size=7, alpha=0.7,
                hue='Time',
                palette={'Pre-test': pre_color, 'Post-test': post_color},
                legend=False)
    
    # Add mean points with error bars
    ax1.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
                fmt='o', color='black', markersize=10, capsize=10, 
                markerfacecolor='white', markeredgewidth=2)
    
    # Add mean values as text
    ax1.text(0, pre_mean + pre_se + 0.2, f'{pre_mean:.2f}', ha='center', fontweight='bold')
    ax1.text(1, post_mean + post_se + 0.2, f'{post_mean:.2f}', ha='center', fontweight='bold')
    
    # Add significance indicator if significant
    y_max = max(pre_efficacy.max(), post_efficacy.max()) + 0.5
    if p_corrected < 0.05:
        stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*"
        ax1.plot([0, 0, 1, 1], [y_max, y_max+0.1, y_max+0.1, y_max], lw=1.5, c='black')
        ax1.text(0.5, y_max+0.2, stars, ha='center', va='bottom', fontsize=16)

    # Customize violin panel
    ax1.set_title('Distribution of Self-Efficacy Scores', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Self-Efficacy Score (1-7 scale)', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_ylim(1, y_max + 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # RIGHT PANEL: Bar chart
    bar_width = 0.7
    x_pos = [0, 1]
    bars = ax2.bar(x_pos, [pre_mean, post_mean], yerr=[pre_se, post_se], 
                  width=bar_width, capsize=10, color=[pre_color, post_color], 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (pre_se if i==0 else post_se) + 0.15,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add significance indicator if significant
    if p_corrected < 0.05:
        stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*"
        y_pos = max(pre_mean + pre_se, post_mean + post_se) + 0.4
        ax2.plot([0, 0, 1, 1], [y_pos-0.1, y_pos, y_pos, y_pos-0.1], 'k-', linewidth=1.5)
        ax2.text(0.5, y_pos + 0.1, stars, ha='center', va='bottom', fontsize=16)
    
    # Add detailed statistics text box - FIXED: p-value format limited to 4 digits
    p_display = f"{p_corrected:.4f}" if p_corrected >= 0.0001 else "<0.0001"
    stats_text = (f"t({len(pre_efficacy)-1}) = {results_df.loc[results_df['Component'] == 'Self-Efficacy', 'T_stat'].values[0]:.2f}\n"
                 f"p = {p_display}\n"
                 f"Cohen's d = {d:.2f}\n"
                 f"Pre: {pre_mean:.2f} ± {pre_sd:.2f}\n"
                 f"Post: {post_mean:.2f} ± {post_sd:.2f}\n"
                 f"Change: +{post_mean - pre_mean:.2f}")
    
    ax2.text(0.5, 4.0, stats_text, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'),
           fontsize=11)
    
    # Customize bar chart panel
    ax2.set_title('Mean Self-Efficacy Scores', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Self-Efficacy Score (1-7 scale)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Pre-test', 'Post-test'], fontsize=12)
    ax2.set_ylim(1, 7.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Overall title
    plt.suptitle('Self-Efficacy in Argumentation', fontsize=16, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('self_efficacy_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig('self_efficacy_combined.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created combined visualization for self-efficacy (violin + bar chart)")



def create_toulmin_density_boxplot(pre_toulmin, post_toulmin):
    """Create density plot for overall Toulmin scores with means and box plot"""
    # Calculate statistics
    pre_mean = pre_toulmin.mean()
    post_mean = post_toulmin.mean()
    pre_sd = pre_toulmin.std()
    post_sd = post_toulmin.std()
    pre_se = pre_sd / np.sqrt(len(pre_toulmin))
    post_se = post_sd / np.sqrt(len(post_toulmin))
    
    # Calculate t-test and effect size
    t_stat, p_value = stats.ttest_rel(post_toulmin, pre_toulmin)
    d = (post_mean - pre_mean) / pre_sd
    
    # Calculate confidence intervals
    ci_low, ci_high = stats.t.interval(0.95, len(pre_toulmin)-1, 
                                     loc=post_mean - pre_mean, 
                                     scale=stats.sem(post_toulmin - pre_toulmin))
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                 gridspec_kw={'width_ratios': [2, 1]})
    
    # Colors
    pre_color = "#4C72B0"  # Blue
    post_color = "#55A868"  # Green
    
    # LEFT PANEL: Clean density plot with no data points
    sns.kdeplot(x=pre_toulmin, ax=ax1, fill=True, 
               label=f'Pre-test (M = {pre_mean:.2f})', color=pre_color, alpha=0.7)
    sns.kdeplot(x=post_toulmin, ax=ax1, fill=True, 
               label=f'Post-test (M = {post_mean:.2f})', color=post_color, alpha=0.7)
    
    # Add mean lines
    ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
    ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
    # Add shaded area for mean difference
    ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
    # Add effect size annotation (value only, no category label)
    effect_size_text = f"d = {d:.2f}"
    
    ax1.text((pre_mean + post_mean)/2, 0.9, effect_size_text, 
            ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add delta indicator (moved higher)
    ax1.text((pre_mean + post_mean)/2, 0.65, f'Δ = {post_mean - pre_mean:.2f}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9))
    
    # Customize left panel
    ax1.set_title('Distribution of Overall Toulmin Scores', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Toulmin Score (1-7 scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(1, 7)
    ax1.set_ylim(0, 1.1)  # Remove white space below 0
    ax1.grid(linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper left')
    
    # RIGHT PANEL: Box plot
    # Prepare data for boxplot
    box_data = [pre_toulmin, post_toulmin]
    
    # Create custom boxplot
    boxprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    medianprops = dict(linewidth=2.5, color='black')
    meanprops = dict(marker='o', markerfacecolor='white', markeredgecolor='black',
                   markersize=8, markeredgewidth=2)
    
    # Create boxplot without showing outliers (we'll add all points manually)
    bplot = ax2.boxplot(box_data, patch_artist=True, 
                      showmeans=True, meanline=False, showfliers=False,
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
    
    # Add individual data points
    for i, data in enumerate([pre_toulmin, post_toulmin]):
        # Add jittered points
        x_pos = i + 1
        y = data.values
        # Add points with slight jitter
        jitter = np.random.normal(0, 0.05, size=len(y))
        ax2.scatter(x_pos + jitter, y, color=pre_color if i == 0 else post_color,
                  alpha=0.7, s=30, edgecolor='white', linewidth=0.5)
    
    # Add mean values as text (placed at the bottom to avoid overlap)
    ax2.text(1, 1.0, f'Mean = {pre_mean:.2f}', ha='center', 
           fontsize=10, fontweight='bold', color=pre_color,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    ax2.text(2, 1.0, f'Mean = {post_mean:.2f}', ha='center', 
           fontsize=10, fontweight='bold', color=post_color,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Add significance line and stars
    y_max = max(pre_toulmin.max(), post_toulmin.max()) + 0.3
    ax2.plot([1, 1, 2, 2], [y_max, y_max+0.1, y_max+0.1, y_max], lw=1.5, c='black')
    
    # Add stars for significance
    stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
    ax2.text(1.5, y_max + 0.2, stars, ha='center', va='bottom', fontsize=16)
    
    # Add statistics box
    stats_text = f"t({len(pre_toulmin)-1}) = {t_stat:.2f}\np = {p_value:.3f}\nd = {d:.2f}"
    ax2.text(1.5, 2.0, stats_text, ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Customize right panel
    ax2.set_title('Box Plot Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Score (1-7 scale)', fontsize=12)
    ax2.set_ylim(0.7, y_max + 0.5)  # Start from lower value to show mean labels
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add overall title
    plt.suptitle('Changes in Six Toulmin Components After Using Mind Elevator', 
               fontsize=16, fontweight='bold', y=0.98)
    
    # Add paired sample t-test results as a footnote
    plt.figtext(0.5, 0.01, 
               f"Paired t-test: t({len(pre_toulmin)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig('toulmin_overall_density_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('toulmin_overall_density_boxplot.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created density plot with box plot for overall Toulmin scores")
    return fig



if __name__ == "__main__":
    results = analyze_argumentation_efficacy('pre_form.csv', 'post_form.csv')
    
    # Extract data for additional visualizations
    pre_df = pd.read_csv('pre_form.csv')
    post_df = pd.read_csv('post_form.csv')
    
    # Extract self-efficacy data
    pre_efficacy = pre_df['self_ability_argument'].astype(float)
    post_efficacy = post_df['self_ability_argument'].astype(float)
    
    # Extract Toulmin components
    toulmin_components = {
        'Claim': 'Claim',
        'Grounds': 'Grounds',
        'Warrant': 'Warrant',
        'Backing': 'Backing',
        'Qualifier': 'Qualifier',
        'Rebuttals': 'Rebuttals'
    }
    
    # Calculate composite Toulmin score
    pre_toulmin = pre_df[list(toulmin_components.values())].mean(axis=1)
    post_toulmin = post_df[list(toulmin_components.values())].mean(axis=1)
    
    # Create additional visualizations
    create_combined_self_efficacy_visualization(pre_efficacy, post_efficacy, results['individual_components'])
    create_improved_toulmin_violin(pre_df, post_df, toulmin_components, results['individual_components'])
    create_toulmin_self_efficacy_correlation(pre_toulmin, post_toulmin, pre_efficacy, post_efficacy)
    create_toulmin_density_boxplot(pre_toulmin, post_toulmin)

























# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from statsmodels.stats.multitest import multipletests
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# import matplotlib.gridspec as gridspec

# def analyze_argumentation_efficacy(pre_csv, post_csv):
#     """
#     Comprehensive analysis of argumentation skills and self-efficacy (RQ3)
#     """
#     # Load data
#     pre_df = pd.read_csv(pre_csv)
#     post_df = pd.read_csv(post_csv)
    
#     print("Analyzing argumentation skills and self-efficacy (RQ3)...\n")
    
#     # Define the specific columns for Toulmin model components
#     toulmin_components = {
#         'Claim': 'Claim',
#         'Grounds': 'Grounds',
#         'Warrant': 'Warrant',
#         'Backing': 'Backing',
#         'Qualifier': 'Qualifier',
#         'Rebuttals': 'Rebuttals'
#     }
    
#     # Define self-efficacy column
#     self_efficacy_col = 'self_ability_argument'
    
#     # Check columns exist in datasets
#     for component_name, col in toulmin_components.items():
#         if col not in pre_df.columns or col not in post_df.columns:
#             print(f"ERROR: '{col}' column not found in one or both datasets")
#             return
    
#     if self_efficacy_col not in pre_df.columns or self_efficacy_col not in post_df.columns:
#         print(f"ERROR: '{self_efficacy_col}' column not found in one or both datasets")
#         return
    
#     # Create dataframe to store all analysis results
#     results_df = pd.DataFrame(columns=[
#         'Component', 'Pre_Mean', 'Pre_SD', 'Post_Mean', 'Post_SD', 
#         'Mean_Diff', 'Percent_Change', 'T_stat', 'P_value', 
#         'Cohens_d', 'CI_Low', 'CI_High', 'Improved_Pct'
#     ])
    
#     # Store p-values for Holm-Bonferroni correction
#     p_values = []
#     component_names = []
    
#     # 1. Process each Toulmin component
#     for component_name, col in toulmin_components.items():
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
#         component_names.append(component_name)
        
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
#             'Component': [component_name],
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
    
#     # 2. Process self-efficacy
#     pre_efficacy = pre_df[self_efficacy_col].astype(float)
#     post_efficacy = post_df[self_efficacy_col].astype(float)
    
#     # Basic statistics for self-efficacy
#     pre_eff_mean = pre_efficacy.mean()
#     pre_eff_sd = pre_efficacy.std()
#     post_eff_mean = post_efficacy.mean()
#     post_eff_sd = post_efficacy.std()
#     eff_mean_diff = post_eff_mean - pre_eff_mean
#     eff_percent_change = (eff_mean_diff / pre_eff_mean) * 100
    
#     # Paired t-test for self-efficacy
#     t_eff, p_eff = stats.ttest_rel(post_efficacy, pre_efficacy)
    
#     # Add self-efficacy p-value to list for correction
#     p_values.append(p_eff)
#     component_names.append('Self-Efficacy')
    
#     # Effect size for self-efficacy
#     d_eff = eff_mean_diff / pre_eff_sd
    
#     # Confidence interval for self-efficacy
#     ci_low_eff, ci_high_eff = stats.t.interval(
#         0.95, len(pre_efficacy)-1, 
#         loc=eff_mean_diff, 
#         scale=stats.sem(post_efficacy - pre_efficacy)
#     )
    
#     # Percentage improved for self-efficacy
#     improved_eff = np.sum(post_efficacy > pre_efficacy)
#     improved_eff_pct = (improved_eff / len(pre_efficacy)) * 100
    
#     # Add self-efficacy to results
#     results_df = pd.concat([results_df, pd.DataFrame({
#         'Component': ['Self-Efficacy'],
#         'Pre_Mean': [pre_eff_mean],
#         'Pre_SD': [pre_eff_sd],
#         'Post_Mean': [post_eff_mean],
#         'Post_SD': [post_eff_sd],
#         'Mean_Diff': [eff_mean_diff],
#         'Percent_Change': [eff_percent_change],
#         'T_stat': [t_eff],
#         'P_value': [p_eff],
#         'Cohens_d': [d_eff],
#         'CI_Low': [ci_low_eff],
#         'CI_High': [ci_high_eff],
#         'Improved_Pct': [improved_eff_pct]
#     })], ignore_index=True)
    
#     # 3. Calculate composite Toulmin score
#     pre_toulmin = pre_df[list(toulmin_components.values())].mean(axis=1)
#     post_toulmin = post_df[list(toulmin_components.values())].mean(axis=1)
    
#     # Statistics for composite Toulmin
#     pre_toulmin_mean = pre_toulmin.mean()
#     pre_toulmin_sd = pre_toulmin.std()
#     post_toulmin_mean = post_toulmin.mean()
#     post_toulmin_sd = post_toulmin.std()
#     toulmin_mean_diff = post_toulmin_mean - pre_toulmin_mean
    
#     # T-test for composite Toulmin
#     t_toulmin, p_toulmin = stats.ttest_rel(post_toulmin, pre_toulmin)
#     d_toulmin = toulmin_mean_diff / pre_toulmin_sd
    
#     # 4. Apply Holm-Bonferroni correction
#     reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
#     results_df['P_corrected'] = p_corrected
#     results_df['Significant'] = reject
    
#     # 5. Calculate correlation between Toulmin and self-efficacy
#     # For pre-test
#     pre_toulmin_efficacy_corr, pre_toulmin_efficacy_p = stats.pearsonr(pre_toulmin, pre_efficacy)
#     # For post-test
#     post_toulmin_efficacy_corr, post_toulmin_efficacy_p = stats.pearsonr(post_toulmin, post_efficacy)
#     # For change scores
#     toulmin_change = post_toulmin - pre_toulmin
#     efficacy_change = post_efficacy - pre_efficacy
#     change_corr, change_corr_p = stats.pearsonr(toulmin_change, efficacy_change)
    
#     # Print detailed results
#     print("\n=== ARGUMENTATION SKILLS & SELF-EFFICACY ANALYSIS (RQ3) ===")
#     print("\nIndividual Components Analysis:")
#     print("-" * 100)
#     print(f"{'Component':<20} {'Pre':<10} {'Post':<10} {'Change':<10} {'t-stat':<10} {'p-value':<10} {'p-corr':<10} {'Sig?':<6} {'d':<8}")
#     print("-" * 100)
    
#     for _, row in results_df.iterrows():
#         sig_symbol = "***" if row['P_corrected'] < 0.001 else "**" if row['P_corrected'] < 0.01 else "*" if row['P_corrected'] < 0.05 else ""
#         print(f"{row['Component']:<20} {row['Pre_Mean']:.2f}±{row['Pre_SD']:.2f} {row['Post_Mean']:.2f}±{row['Post_SD']:.2f} {row['Mean_Diff']:+.2f} {row['T_stat']:.2f} {row['P_value']:.4f} {row['P_corrected']:.4f} {sig_symbol:<6} {row['Cohens_d']:.2f}")
    
#     print("-" * 100)
#     print("\nComposite Scores Analysis:")
#     print(f"Overall Toulmin score: {pre_toulmin_mean:.2f}±{pre_toulmin_sd:.2f} → {post_toulmin_mean:.2f}±{post_toulmin_sd:.2f}, t={t_toulmin:.2f}, p={p_toulmin:.4f}, d={d_toulmin:.2f}")
    
#     print("\nCorrelation Analysis (Toulmin vs. Self-Efficacy):")
#     print(f"Pre-test correlation: r = {pre_toulmin_efficacy_corr:.2f}, p = {pre_toulmin_efficacy_p:.4f}")
#     print(f"Post-test correlation: r = {post_toulmin_efficacy_corr:.2f}, p = {post_toulmin_efficacy_p:.4f}")
#     print(f"Change scores correlation: r = {change_corr:.2f}, p = {change_corr_p:.4f}")
    
#     # Create visualizations
#     print("\nCreating visualizations...")
    
#     # 1. Enhanced visualizations for argumentation analysis
#     create_individual_violin_charts(pre_df, post_df, toulmin_components, results_df)
#     create_self_efficacy_violin(pre_efficacy, post_efficacy, results_df)
#     create_improved_toulmin_violin(pre_df, post_df, toulmin_components, results_df)
#     create_toulmin_density_with_means(pre_toulmin, post_toulmin)
#     create_toulmin_self_efficacy_scatter(pre_toulmin, post_toulmin, pre_efficacy, post_efficacy)
#     create_toulmin_radar_chart(results_df.iloc[:6], "Toulmin Model Components: Pre vs Post")
#     create_bar_chart(results_df, "Changes in Argumentation Skills & Self-Efficacy")
#     visualize_holm_bonferroni(results_df)
    
#     # Save results to CSV
#     results_df.to_csv('argumentation_analysis_results.csv', index=False)
#     print("\nResults saved to 'argumentation_analysis_results.csv'")
    
#     # Store comprehensive results for return
#     return {
#         'individual_components': results_df,
#         'composite_toulmin': {
#             'pre_mean': pre_toulmin_mean,
#             'pre_sd': pre_toulmin_sd,
#             'post_mean': post_toulmin_mean,
#             'post_sd': post_toulmin_sd,
#             't_stat': t_toulmin,
#             'p_value': p_toulmin,
#             'cohens_d': d_toulmin,
#             'pre_data': pre_toulmin,
#             'post_data': post_toulmin
#         },
#         'self_efficacy': {
#             'pre_mean': pre_eff_mean,
#             'pre_sd': pre_eff_sd,
#             'post_mean': post_eff_mean,
#             'post_sd': post_eff_sd,
#             't_stat': t_eff,
#             'p_value': p_eff,
#             'cohens_d': d_eff,
#             'pre_data': pre_efficacy,
#             'post_data': post_efficacy
#         },
#         'correlations': {
#             'pre_correlation': pre_toulmin_efficacy_corr,
#             'pre_p_value': pre_toulmin_efficacy_p,
#             'post_correlation': post_toulmin_efficacy_corr,
#             'post_p_value': post_toulmin_efficacy_p,
#             'change_correlation': change_corr,
#             'change_p_value': change_corr_p
#         }
#     }

# def create_individual_violin_charts(pre_df, post_df, components, results_df):
#     """Create separate IEEE-friendly violin charts for each argumentation component"""
#     for component_name, col in components.items():
#         # Extract data
#         pre_data = pre_df[col].astype(float)
#         post_data = post_df[col].astype(float)
        
#         # Prepare data for plotting
#         data_long = pd.DataFrame({
#             'Time': ['Pre-test'] * len(pre_data) + ['Post-test'] * len(post_data),
#             'Score': np.concatenate([pre_data, post_data])
#         })
        
#         # Calculate statistics
#         pre_mean = pre_data.mean()
#         pre_sd = pre_data.std()
#         post_mean = post_data.mean()
#         post_sd = post_data.std()
#         pre_se = pre_sd / np.sqrt(len(pre_data))
#         post_se = post_sd / np.sqrt(len(post_data))
        
#         # Paired t-test
#         t_stat, p_value = stats.ttest_rel(post_data, pre_data)
#         d = (post_mean - pre_mean) / pre_sd
        
#         # Get corrected p-value from results_df
#         p_corrected = results_df.loc[results_df['Component'] == component_name, 'P_corrected'].values[0]
        
#         # Create figure
#         plt.figure(figsize=(7, 8))
        
#         # Create violin plot (with clear separation)
#         ax = sns.violinplot(x='Time', y='Score', data=data_long, 
#                          inner=None, 
#                          hue='Time',
#                          palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'},
#                          legend=False)
        
#         # Add individual data points
#         sns.stripplot(x='Time', y='Score', data=data_long, 
#                   jitter=True, size=6, alpha=0.7,
#                   hue='Time',
#                   palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'},
#                   legend=False)
        
#         # Add mean points with error bars
#         plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
#                    fmt='o', color='black', markersize=10, capsize=10, 
#                    markerfacecolor='white', markeredgewidth=2, elinewidth=2)
        
#         # Add mean values as text
#         plt.text(0, pre_mean + pre_se + 0.2, f'M = {pre_mean:.2f}', ha='center', fontweight='bold')
#         plt.text(1, post_mean + post_se + 0.2, f'M = {post_mean:.2f}', ha='center', fontweight='bold')
        
#         # Add significance indicators with Holm-Bonferroni note
#         max_y = max(pre_data.max(), post_data.max()) + 0.5
#         if p_corrected < 0.05:
#             stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*"
#             plt.plot([0, 0, 1, 1], [max_y, max_y+0.1, max_y+0.1, max_y], lw=1.5, c='black')
#             plt.text(0.5, max_y+0.2, stars, ha='center', va='bottom', fontsize=16)
            
#             # Add statistics text in box
#             stats_text = f"t({len(pre_data)-1}) = {t_stat:.2f}\np = {p_corrected:.3f}, d = {d:.2f}"
#             plt.text(0.5, 2.0, stats_text, ha='center', va='center',
#                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
#         # Set title and labels
#         plt.title(f'Toulmin Component: {component_name}', fontsize=15, pad=20)
#         plt.ylabel('Score (1-7 scale)', fontsize=12)
#         plt.xlabel('', fontsize=12)
#         plt.ylim(0.5, max_y+1.0)
        
#         # Add grid lines
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
        
#         # Add Holm-Bonferroni note
#         plt.figtext(0.5, 0.01, "Note: p-values corrected using Holm-Bonferroni method", 
#                   ha="center", fontsize=9, style='italic')
        
#         # Save figure
#         plt.tight_layout()
#         filename = f"violin_toulmin_{component_name.lower()}"
#         plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
#         plt.savefig(f'{filename}.pdf', bbox_inches='tight')
#         plt.close()
        
#     print("Created IEEE-friendly violin charts for each Toulmin component")

# def create_self_efficacy_violin(pre_efficacy, post_efficacy, results_df):
#     """Create violin chart for self-efficacy in argumentation"""
#     # Prepare data for plotting
#     data_long = pd.DataFrame({
#         'Time': ['Pre-test'] * len(pre_efficacy) + ['Post-test'] * len(post_efficacy),
#         'Self-Efficacy Score': np.concatenate([pre_efficacy, post_efficacy])
#     })
    
#     # Calculate statistics
#     pre_mean = pre_efficacy.mean()
#     pre_sd = pre_efficacy.std()
#     post_mean = post_efficacy.mean()
#     post_sd = post_efficacy.std()
#     pre_se = pre_sd / np.sqrt(len(pre_efficacy))
#     post_se = post_sd / np.sqrt(len(post_efficacy))
    
#     # Paired t-test
#     t_stat, p_value = stats.ttest_rel(post_efficacy, pre_efficacy)
#     d = (post_mean - pre_mean) / pre_sd
    
#     # Get corrected p-value from results_df
#     p_corrected = results_df.loc[results_df['Component'] == 'Self-Efficacy', 'P_corrected'].values[0]
    
#     # Create figure
#     plt.figure(figsize=(8, 9))
    
#     # Create violin plot
#     ax = sns.violinplot(x='Time', y='Self-Efficacy Score', data=data_long, 
#                       inner=None, 
#                       hue='Time',
#                       palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'},
#                       legend=False)
    
#     # Add scatter points with jitter
#     sns.stripplot(x='Time', y='Self-Efficacy Score', data=data_long, 
#                 jitter=True, size=7, alpha=0.7,
#                 hue='Time',
#                 palette={'Pre-test': '#4C72B0', 'Post-test': '#55A868'},
#                 legend=False)
    
#     # Add mean points with error bars
#     plt.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se], 
#                 fmt='o', color='black', markersize=10, capsize=10, 
#                 markerfacecolor='white', markeredgewidth=2, elinewidth=2)
    
#     # Add mean values as text
#     plt.text(0, pre_mean + pre_se + 0.2, f'M = {pre_mean:.2f}', ha='center', fontweight='bold')
#     plt.text(1, post_mean + post_se + 0.2, f'M = {post_mean:.2f}', ha='center', fontweight='bold')
    
#     # Add significance annotation
#     max_y = max(pre_efficacy.max(), post_efficacy.max()) + 0.5
#     if p_corrected < 0.05:
#         stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*"
#         plt.plot([0, 0, 1, 1], [max_y, max_y+0.1, max_y+0.1, max_y], lw=1.5, c='black')
#         plt.text(0.5, max_y+0.2, stars, ha='center', va='bottom', fontsize=16)
        
#         # Add statistics box
#         stats_text = f"t({len(pre_efficacy)-1}) = {t_stat:.2f}\np = {p_corrected:.3f}\nd = {d:.2f}"
#         plt.text(0.5, 2.0, stats_text, ha='center', va='center',
#                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
#     # Set title and labels
#     plt.title('Self-Efficacy in Argumentation', fontsize=15, pad=20)
#     plt.ylabel('Score (1-7 scale)', fontsize=12)
#     plt.xlabel('', fontsize=12)
#     plt.ylim(1.0, max_y+1.0)
    
#     # Add grid lines
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # Add Holm-Bonferroni note
#     plt.figtext(0.5, 0.01, "Note: p-value corrected using Holm-Bonferroni method", 
#               ha="center", fontsize=9, style='italic')
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig('self_efficacy_violin.png', dpi=300, bbox_inches='tight')
#     plt.savefig('self_efficacy_violin.pdf', bbox_inches='tight')
#     plt.close()
    
#     print("Created violin plot for self-efficacy in argumentation")

# def create_improved_toulmin_violin(pre_df, post_df, components, results_df):
#     """Create an improved, clean violin plot with all Toulmin components"""
#     # Set up the figure
#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
#     axes = axes.flatten()
    
#     # Colors
#     pre_color = '#4C72B0'  # Blue
#     post_color = '#55A868'  # Green
    
#     # Process each component
#     for i, (component_name, col) in enumerate(components.items()):
#         # Extract data
#         pre_data = pre_df[col].astype(float)
#         post_data = post_df[col].astype(float)
        
#         # Put into long format for seaborn
#         data = pd.DataFrame({
#             'Pre-test': pre_data,
#             'Post-test': post_data
#         })
#         data_long = data.melt(var_name='Time', value_name='Score')
        
#         # Calculate statistics
#         pre_mean = pre_data.mean()
#         pre_sd = pre_data.std()
#         post_mean = post_data.mean()
#         post_sd = post_data.std()
#         pre_se = pre_sd / np.sqrt(len(pre_data))
#         post_se = post_sd / np.sqrt(len(post_data))
        
#         # Get corrected p-value
#         p_corr = results_df.loc[results_df['Component'] == component_name, 'P_corrected'].values[0]
        
#         # Create violin plot on the appropriate subplot
#         ax = axes[i]
#         sns.violinplot(x='Time', y='Score', data=data_long, ax=ax, 
#                       inner=None, 
#                       palette={'Pre-test': pre_color, 'Post-test': post_color})
        
 
#         # Add mean markers
#         ax.errorbar([0, 1], [pre_mean, post_mean], yerr=[pre_se, post_se],
#                   fmt='o', color='black', markersize=8, capsize=6, 
#                   markerfacecolor='white', markeredgewidth=1.5)
        
#         # Add mean values as text
#         ax.text(0, pre_mean + pre_se + 0.15, f'{pre_mean:.2f}', ha='center', fontsize=10, fontweight='bold')
#         ax.text(1, post_mean + post_se + 0.15, f'{post_mean:.2f}', ha='center', fontsize=10, fontweight='bold')
        
#         # Add significance stars if significant
#         if p_corr < 0.05:
#             stars = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*"
#             y_pos = max(pre_mean + pre_se, post_mean + post_se) + 0.5
#             ax.plot([0, 0, 1, 1], [y_pos, y_pos+0.1, y_pos+0.1, y_pos], color='black', lw=1.2)
#             ax.text(0.5, y_pos+0.15, stars, ha='center', fontsize=14)
        
#         # Set title and labels
#         ax.set_title(component_name, fontsize=14, fontweight='bold')
#         ax.set_ylabel('Score (1-7 scale)' if i % 3 == 0 else '')
#         ax.set_xlabel('')
#         ax.set_ylim(1, 7.5)
        
#         # Add grid
#         ax.grid(axis='y', linestyle='--', alpha=0.4)
    

    
#     # Save figure
#     plt.savefig('toulmin_components_violin.png', dpi=300, bbox_inches='tight')
#     plt.savefig('toulmin_components_violin.pdf', bbox_inches='tight')
#     plt.close()
    
#     print("Created improved visualization of Toulmin components")

# def create_toulmin_density_with_means(pre_toulmin, post_toulmin):
#     """Create density plot for overall Toulmin scores with means"""
#     # Calculate statistics
#     pre_mean = pre_toulmin.mean()
#     post_mean = post_toulmin.mean()
#     pre_sd = pre_toulmin.std()
#     post_sd = post_toulmin.std()
#     pre_se = pre_sd / np.sqrt(len(pre_toulmin))
#     post_se = post_sd / np.sqrt(len(post_toulmin))
    
#     # Calculate t-test and effect size
#     t_stat, p_value = stats.ttest_rel(post_toulmin, pre_toulmin)
#     d = (post_mean - pre_mean) / pre_sd
    
#     # Calculate confidence intervals
#     ci_low, ci_high = stats.t.interval(0.95, len(pre_toulmin)-1, 
#                                      loc=post_mean - pre_mean, 
#                                      scale=stats.sem(post_toulmin - pre_toulmin))
    
#     # Create figure with two panels
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
#                                  gridspec_kw={'width_ratios': [2, 1]})
    
#     # Colors
#     pre_color = "#4C72B0"  # Blue
#     post_color = "#55A868"  # Green
    
#     # Left panel: Density plot
#     sns.kdeplot(x=pre_toulmin, ax=ax1, fill=True, 
#                label=f'Pre-test (M = {pre_mean:.2f})', color=pre_color, alpha=0.7)
#     sns.kdeplot(x=post_toulmin, ax=ax1, fill=True, 
#                label=f'Post-test (M = {post_mean:.2f})', color=post_color, alpha=0.7)
    
#     # Add mean lines
#     ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
#     ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
#     # Add shaded area for mean difference
#     ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
#     # Add effect size annotation
#     effect_size_text = f"d = {d:.2f} (huge effect)"
#     if d < 0.8:
#         effect_size_text = f"d = {d:.2f} (medium effect)" if d >= 0.5 else f"d = {d:.2f} (small effect)"
    
#     ax1.text((pre_mean + post_mean)/2, 0.85, effect_size_text, 
#             ha='center', va='center', fontweight='bold',
#             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
#     # Add delta indicator
#     ax1.text((pre_mean + post_mean)/2, 0.4, f'Δ = {post_mean - pre_mean:.2f}', 
#             ha='center', va='center', fontsize=12, fontweight='bold',
#             bbox=dict(facecolor='white', alpha=0.9))
    
#     # Plot all data points at the bottom
#     y_pos_pre = np.full(len(pre_toulmin), -0.05)
#     y_pos_post = np.full(len(post_toulmin), -0.15)
    
#     ax1.scatter(pre_toulmin, y_pos_pre, color=pre_color, s=30, alpha=0.7, 
#               edgecolor='white', linewidth=0.5, label=None)
#     ax1.scatter(post_toulmin, y_pos_post, color=post_color, s=30, alpha=0.7, 
#               edgecolor='white', linewidth=0.5, label=None)
    
#     # Add labels for the data points
#     center_x = (pre_mean + post_mean)/2
#     ax1.text(center_x - 0.1, -0.05, "Pre-test scores", color=pre_color, 
#            ha='right', va='center', fontsize=10)
#     ax1.text(center_x + 0.1, -0.15, "Post-test scores", color=post_color, 
#            ha='left', va='center', fontsize=10)
    
#     # Customize left panel
#     ax1.set_title('Distribution of Overall Toulmin Scores', fontsize=14, fontweight='bold', pad=15)
#     ax1.set_xlabel('Toulmin Score (1-7 scale)', fontsize=12)
#     ax1.set_ylabel('Density', fontsize=12)
#     ax1.set_xlim(1, 7)
#     ax1.set_ylim(-0.2, 1.2)
#     ax1.grid(linestyle='--', alpha=0.6)
#     ax1.legend(fontsize=11, loc='upper left')
    
#     # Right panel: Bar chart
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
    
#     # Add significance line and stars
#     y_max = max(pre_mean + pre_se, post_mean + post_se) + 0.3
#     ax2.plot([0, 0, 1, 1], [y_max-0.1, y_max, y_max, y_max-0.1], lw=1.5, c='black')
    
#     # Add stars for significance
#     stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
#     ax2.text(0.5, y_max + 0.1, stars, ha='center', va='bottom', fontsize=16)
    
#     # Customize right panel
#     ax2.set_title('Mean Overall Toulmin Score', fontsize=14, fontweight='bold', pad=15)
#     ax2.set_ylabel('Score (1-7 scale)', fontsize=12)
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels(['Pre-test', 'Post-test'], fontsize=11)
#     ax2.set_ylim(1, y_max + 0.5)
#     ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
#     # Add overall title
#     plt.suptitle('Changes in Six Toulmin Components After Using Mind Elevator', 
#                fontsize=16, fontweight='bold', y=0.98)
    
#     # Add paired sample t-test results as a footnote
#     plt.figtext(0.5, 0.01, 
#                f"Paired t-test: t({len(pre_toulmin)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
#                ha='center', fontsize=10, style='italic')
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save figure
#     plt.savefig('toulmin_overall_density.png', dpi=300, bbox_inches='tight')
#     plt.savefig('toulmin_overall_density.pdf', bbox_inches='tight')
#     plt.close()
    
#     print("Created density plot with means for overall Toulmin scores")


def create_toulmin_density_boxplot(pre_toulmin, post_toulmin):
    """Create density plot for overall Toulmin scores with means and box plot"""
    # Calculate statistics
    pre_mean = pre_toulmin.mean()
    post_mean = post_toulmin.mean()
    pre_sd = pre_toulmin.std()
    post_sd = post_toulmin.std()
    pre_se = pre_sd / np.sqrt(len(pre_toulmin))
    post_se = post_sd / np.sqrt(len(post_toulmin))
    
    # Calculate t-test and effect size
    t_stat, p_value = stats.ttest_rel(post_toulmin, pre_toulmin)
    d = (post_mean - pre_mean) / pre_sd
    
    # Calculate confidence intervals
    ci_low, ci_high = stats.t.interval(0.95, len(pre_toulmin)-1, 
                                     loc=post_mean - pre_mean, 
                                     scale=stats.sem(post_toulmin - pre_toulmin))
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                 gridspec_kw={'width_ratios': [2, 1]})
    
    # Colors
    pre_color = "#4C72B0"  # Blue
    post_color = "#55A868"  # Green
    
    # LEFT PANEL: Clean density plot with no data points
    sns.kdeplot(x=pre_toulmin, ax=ax1, fill=True, 
               label=f'Pre-test (M = {pre_mean:.2f})', color=pre_color, alpha=0.7)
    sns.kdeplot(x=post_toulmin, ax=ax1, fill=True, 
               label=f'Post-test (M = {post_mean:.2f})', color=post_color, alpha=0.7)
    
    # Add mean lines
    ax1.axvline(pre_mean, color=pre_color, linestyle='--', linewidth=2)
    ax1.axvline(post_mean, color=post_color, linestyle='--', linewidth=2)
    
    # Add shaded area for mean difference
    ax1.axvspan(pre_mean, post_mean, alpha=0.1, color='gray')
    
    # Add delta indicator only (NO effect size annotation)
    ax1.text((pre_mean + post_mean)/2, 0.75, f'Δ = {post_mean - pre_mean:.2f}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9))
    
    # Customize left panel
    ax1.set_title('Distribution of Overall Toulmin Scores', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Toulmin Score (1-7 scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xlim(1, 7)
    ax1.set_ylim(0, 1.1)  # Remove white space below 0
    ax1.grid(linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper left')
    
    # RIGHT PANEL: Box plot
    # Prepare data for boxplot
    box_data = [pre_toulmin, post_toulmin]
    
    # Create custom boxplot
    boxprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    medianprops = dict(linewidth=2.5, color='black')
    meanprops = dict(marker='o', markerfacecolor='white', markeredgecolor='black',
                   markersize=8, markeredgewidth=2)
    
    # Create boxplot without showing outliers (we'll add all points manually)
    bplot = ax2.boxplot(box_data, patch_artist=True, 
                      showmeans=True, meanline=False, showfliers=False,
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
    
    # Add individual data points
    for i, data in enumerate([pre_toulmin, post_toulmin]):
        # Add jittered points
        x_pos = i + 1
        y = data.values
        # Add points with slight jitter
        jitter = np.random.normal(0, 0.05, size=len(y))
        ax2.scatter(x_pos + jitter, y, color=pre_color if i == 0 else post_color,
                  alpha=0.7, s=30, edgecolor='white', linewidth=0.5)
    
    # Add mean values as text (placed at the bottom to avoid overlap)
    ax2.text(1, 1.0, f'Mean = {pre_mean:.2f}', ha='center', 
           fontsize=10, fontweight='bold', color=pre_color,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    ax2.text(2, 1.0, f'Mean = {post_mean:.2f}', ha='center', 
           fontsize=10, fontweight='bold', color=post_color,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Add significance line and stars
    y_max = max(pre_toulmin.max(), post_toulmin.max()) + 0.3
    ax2.plot([1, 1, 2, 2], [y_max, y_max+0.1, y_max+0.1, y_max], lw=1.5, c='black')
    
    # Add stars for significance
    stars = '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
    ax2.text(1.5, y_max + 0.2, stars, ha='center', va='bottom', fontsize=16)
    
    # Add statistics box
    stats_text = f"t({len(pre_toulmin)-1}) = {t_stat:.2f}\np = {p_value:.3f}\nd = {d:.2f}"
    ax2.text(1.5, 2.0, stats_text, ha='center', va='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Customize right panel
    ax2.set_title('Box Plot Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Score (1-7 scale)', fontsize=12)
    ax2.set_ylim(0.7, y_max + 0.5)  # Start from lower value to show mean labels
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add overall title
    plt.suptitle('Changes in Six Toulmin Components After Using Mind Elevator', 
               fontsize=16, fontweight='bold', y=0.98)
    
    # Add paired sample t-test results as a footnote
    plt.figtext(0.5, 0.01, 
               f"Paired t-test: t({len(pre_toulmin)-1}) = {t_stat:.2f}, p < 0.001, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig('toulmin_overall_density_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('toulmin_overall_density_boxplot.pdf', bbox_inches='tight')
    plt.close()
    
    print("Created density plot with box plot for overall Toulmin scores")
    return fig

# def create_toulmin_self_efficacy_scatter(pre_toulmin, post_toulmin, pre_efficacy, post_efficacy):
#     """Create scatter plot showing correlation between Toulmin scores and self-efficacy"""
#     # Calculate correlations
#     pre_corr, pre_p = stats.pearsonr(pre_toulmin, pre_efficacy)
#     post_corr, post_p = stats.pearsonr(post_toulmin, post_efficacy)
    
#     # Calculate change scores and their correlation
#     toulmin_change = post_toulmin - pre_toulmin
#     efficacy_change = post_efficacy - pre_efficacy
#     change_corr, change_p = stats.pearsonr(toulmin_change, efficacy_change)
    
#     # Create a figure with 3 subplots
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # Colors
#     pre_color = "#4C72B0"  # Blue
#     post_color = "#55A868"  # Green
#     change_color = "#C44E52"  # Red
    
#     # Plot 1: Pre-test correlation
#     axes[0].scatter(pre_toulmin, pre_efficacy, color=pre_color, s=60, alpha=0.7, 
#                    edgecolor='white', linewidth=0.5)
    
#     # Add best fit line
#     m, b = np.polyfit(pre_toulmin, pre_efficacy, 1)
#     x_line = np.array([min(pre_toulmin), max(pre_toulmin)])
#     axes[0].plot(x_line, m * x_line + b, color=pre_color, linestyle='--', linewidth=2)
    
#     # Add correlation text
#     sig_text = "" if pre_p >= 0.05 else "*" if pre_p < 0.05 else "**" if pre_p < 0.01 else "***"
#     axes[0].text(0.05, 0.95, f"r = {pre_corr:.2f}{sig_text}\np = {pre_p:.3f}", 
#                 transform=axes[0].transAxes, ha='left', va='top', fontsize=12,
#                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
#     # Customize plot
#     axes[0].set_title('Pre-test Correlation', fontsize=14, fontweight='bold')
#     axes[0].set_xlabel('Overall Toulmin Score', fontsize=12)
#     axes[0].set_ylabel('Self-Efficacy Score', fontsize=12)
#     axes[0].grid(linestyle='--', alpha=0.4)
#     axes[0].set_xlim(1, 7)
#     axes[0].set_ylim(1, 7)
    
#     # Plot 2: Post-test correlation
#     axes[1].scatter(post_toulmin, post_efficacy, color=post_color, s=60, alpha=0.7, 
#                    edgecolor='white', linewidth=0.5)
    
#     # Add best fit line
#     m, b = np.polyfit(post_toulmin, post_efficacy, 1)
#     x_line = np.array([min(post_toulmin), max(post_toulmin)])
#     axes[1].plot(x_line, m * x_line + b, color=post_color, linestyle='--', linewidth=2)
    
#     # Add correlation text
#     sig_text = "" if post_p >= 0.05 else "*" if post_p < 0.05 else "**" if post_p < 0.01 else "***"
#     axes[1].text(0.05, 0.95, f"r = {post_corr:.2f}{sig_text}\np = {post_p:.3f}", 
#                 transform=axes[1].transAxes, ha='left', va='top', fontsize=12,
#                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
#     # Customize plot
#     axes[1].set_title('Post-test Correlation', fontsize=14, fontweight='bold')
#     axes[1].set_xlabel('Overall Toulmin Score', fontsize=12)
#     axes[1].set_ylabel('', fontsize=12)
#     axes[1].grid(linestyle='--', alpha=0.4)
#     axes[1].set_xlim(1, 7)
#     axes[1].set_ylim(1, 7)
    
#     # Plot 3: Change score correlation
#     axes[2].scatter(toulmin_change, efficacy_change, color=change_color, s=60, alpha=0.7, 
#                    edgecolor='white', linewidth=0.5)
    
#     # Add best fit line
#     m, b = np.polyfit(toulmin_change, efficacy_change, 1)
#     x_line = np.array([min(toulmin_change), max(toulmin_change)])
#     axes[2].plot(x_line, m * x_line + b, color=change_color, linestyle='--', linewidth=2)
    
#     # Add correlation text
#     sig_text = "" if change_p >= 0.05 else "*" if change_p < 0.05 else "**" if change_p < 0.01 else "***"
#     axes[2].text(0.05, 0.95, f"r = {change_corr:.2f}{sig_text}\np = {change_p:.3f}", 
#                 transform=axes[2].transAxes, ha='left', va='top', fontsize=12,
#                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
#     # Customize plot
#     axes[2].set_title('Change Score Correlation', fontsize=14, fontweight='bold')
#     axes[2].set_xlabel('Change in Toulmin Score', fontsize=12)
#     axes[2].set_ylabel('', fontsize=12)
#     axes[2].grid(linestyle='--', alpha=0.4)
#     axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
#     axes[2].axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
#     # Add overall title
#     plt.suptitle('Relationship Between Argumentation Skills and Self-Efficacy', 
#                fontsize=16, fontweight='bold', y=0.98)
    
#     # Add significance legend at the bottom
#     plt.figtext(0.5, 0.01, '* p < 0.05, ** p < 0.01, *** p < 0.001', 
#               ha='center', fontsize=10, style='italic')
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save figure
#     plt.savefig('toulmin_self_efficacy_correlation.png', dpi=300, bbox_inches='tight')
#     plt.savefig('toulmin_self_efficacy_correlation.pdf', bbox_inches='tight')
#     plt.close()
    
#     print("Created correlation analysis between Toulmin scores and self-efficacy")

# def create_toulmin_radar_chart(results_df, title):
#     """Create improved radar chart with non-overlapping text labels for Toulmin components"""
#     # Number of variables
#     categories = results_df['Component'].tolist()
#     N = len(categories)
    
#     # Pre and post means
#     pre_means = results_df['Pre_Mean'].tolist()
#     post_means = results_df['Post_Mean'].tolist()
    
#     # Calculate angle for each category
#     angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
#     # Close the plot (append first value to end)
#     pre_means_closed = pre_means + [pre_means[0]]
#     post_means_closed = post_means + [post_means[0]]
#     angles_closed = angles + [angles[0]]
    
#     # Create figure with cleaner styling
#     fig = plt.figure(figsize=(10, 10), facecolor='white')
#     ax = fig.add_subplot(111, polar=True)
    
#     # Set background color
#     ax.set_facecolor('#f9f9f9')
    
#     # Add grid lines with subtle styling
#     ax.grid(color='gray', alpha=0.2, linestyle='-')
    
#     # Plot data with thicker lines and larger markers
#     ax.plot(angles_closed, pre_means_closed, 'o-', linewidth=2.5, label='Pre-test', 
#           color='#4C72B0', markersize=8)
#     ax.fill(angles_closed, pre_means_closed, alpha=0.1, color='#4C72B0')
    
#     ax.plot(angles_closed, post_means_closed, 'o-', linewidth=2.5, label='Post-test', 
#           color='#55A868', markersize=8)
#     ax.fill(angles_closed, post_means_closed, alpha=0.1, color='#55A868')
    
#     # Set ticks and limits
#     ax.set_xticks(angles)
#     ax.set_ylim(0, 7)
    
#     # Define custom positions for each label to avoid overlapping
#     label_positions = {
#         'Claim': {'distance': 9.0, 'offset': (0, 0)},
#         'Grounds': {'distance': 9.0, 'offset': (0.3, 0)},
#         'Warrant': {'distance': 9.0, 'offset': (0, 0)},
#         'Backing': {'distance': 9.0, 'offset': (-0.3, 0)},
#         'Qualifier': {'distance': 9.0, 'offset': (0, 0)},
#         'Rebuttals': {'distance': 9.0, 'offset': (0, 0)},
#     }
    
#     # Add labels with custom positioning
#     for i, angle in enumerate(angles):
#         skill = categories[i]
#         position = label_positions.get(skill, {'distance': 8.5, 'offset': (0, 0)})
        
#         # Calculate position
#         x = angle
#         offset_x, offset_y = position['offset']
        
#         # Add the label with background box for clarity
#         ax.text(x + offset_x, position['distance'] + offset_y, skill, 
#               ha='center', va='center', fontsize=12, fontweight='bold', 
#               bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', 
#                        edgecolor='lightgray'))
        
#         # Add mean values directly on the data points
#         # For pre-test
#         pre_x = np.cos(angle) * (pre_means[i])
#         pre_y = np.sin(angle) * (pre_means[i])
#         ax.text(angle, pre_means[i] + 0.2, f"{pre_means[i]:.2f}", 
#               color='#4C72B0', ha='center', va='center', fontsize=10, 
#               fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, 
#                                          boxstyle='round,pad=0.2', edgecolor='#4C72B0'))
        
#         # For post-test
#         post_x = np.cos(angle) * (post_means[i])
#         post_y = np.sin(angle) * (post_means[i])
#         ax.text(angle, post_means[i] + 0.2, f"{post_means[i]:.2f}", 
#               color='#55A868', ha='center', va='center', fontsize=10, 
#               fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, 
#                                          boxstyle='round,pad=0.2', edgecolor='#55A868'))
    
#     # Remove default radial labels and replace with cleaner design
#     ax.set_xticklabels([])
    
#     # Set radial ticks and labels with better styling
#     ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
#     ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7"], fontsize=9, color="gray")
#     ax.set_rlabel_position(0)  # Move radial labels to 0°
    
#     # Add circular gridlines with labels
#     for y in [1, 2, 3, 4, 5, 6, 7]:
#         ax.text(np.pi/4, y, str(y), ha='center', va='center', color='gray', fontsize=9)
    
#     # Add legend with professional styling
#     legend = plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), fontsize=12,
#                       frameon=True, framealpha=0.9, borderpad=1, 
#                       edgecolor='lightgray')
    
#     # Add title with professional styling
#     plt.title(title, size=18, y=1.08, fontweight='bold', color='#333333')
    
#     # Save figure with high quality
#     plt.tight_layout()
#     plt.savefig('toulmin_radar_chart_improved.png', dpi=300, bbox_inches='tight')
#     plt.savefig('toulmin_radar_chart_improved.pdf', bbox_inches='tight')
#     plt.close()
    
#     print("Created improved radar chart with non-overlapping labels")

# def create_bar_chart(results_df, title):
#     """Create improved bar chart with error bars and significance indicators"""
#     # Setup
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     # Data
#     components = results_df['Component'].tolist()
#     pre_means = results_df['Pre_Mean'].tolist()
#     post_means = results_df['Post_Mean'].tolist()
#     pre_sds = results_df['Pre_SD'].tolist()
#     post_sds = results_df['Post_SD'].tolist()
    
#     # Calculate standard errors
#     n = 16  # Number of participants
#     pre_se = [sd / np.sqrt(n) for sd in pre_sds]
#     post_se = [sd / np.sqrt(n) for sd in post_sds]
    
#     # X positions
#     x = np.arange(len(components))
#     width = 0.35
    
#     # Create bars
#     pre_bars = ax.bar(x - width/2, pre_means, width, yerr=pre_se, 
#                     label='Pre-test', color='#4C72B0', capsize=5, 
#                     edgecolor='black', linewidth=1)
#     post_bars = ax.bar(x + width/2, post_means, width, yerr=post_se, 
#                      label='Post-test', color='#55A868', capsize=5, 
#                      edgecolor='black', linewidth=1)
    
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
#             stars = "***" if row['P_corrected'] < 0.001 else "**" if row['P_corrected'] < 0.01 else "*"
                
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
#     ax.set_xticklabels(components, rotation=15, ha='right', fontsize=10)
    
#     # Add dual legends - one for bars and one for significance
#     bar_legend = ax.legend(fontsize=12, loc='upper left')
    
#     # Add significance legend
#     sig_legend_elements = [
#         Line2D([0], [0], marker='', color='none', label='Significance:'),
#         Line2D([0], [0], marker='', color='none', label='* p < 0.05'),
#         Line2D([0], [0], marker='', color='none', label='** p < 0.01'),
#         Line2D([0], [0], marker='', color='none', label='*** p < 0.001')
#     ]
#     sig_legend = ax.legend(handles=sig_legend_elements, loc='upper right', frameon=False)
    
#     # Add both legends
#     ax.add_artist(bar_legend)
#     ax.add_artist(sig_legend)
    
#     # Add note about Holm-Bonferroni
#     plt.figtext(0.5, 0.01, "Note: p-values corrected using Holm-Bonferroni method", 
#                ha="center", fontsize=9, style='italic')
    
#     # Set y-axis to start from 1 (not 0)
#     ax.set_ylim(1, 7.5)
    
#     # Add grid lines
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
    
#     # Save figure
#     plt.tight_layout(rect=[0, 0.03, 1, 0.98])
#     plt.savefig('toulmin_bar_chart_improved.png', dpi=300, bbox_inches='tight')
#     plt.savefig('toulmin_bar_chart_improved.pdf', bbox_inches='tight')
#     plt.close()
    
#     print("Created improved bar chart for all components")

# def visualize_holm_bonferroni(results_df):
#     """Create visualization and explanation of Holm-Bonferroni correction"""
#     # Extract data
#     components = results_df['Component'].tolist()
#     p_values = results_df['P_value'].tolist()
#     p_corrected = results_df['P_corrected'].tolist()
#     is_significant = results_df['Significant'].tolist()
    
#     # Create figure
#     plt.figure(figsize=(10, 7))
    
#     # Sort by p-value for proper Holm-Bonferroni visualization
#     sorted_indices = np.argsort(p_values)
#     sorted_components = [components[i] for i in sorted_indices]
#     sorted_p = [p_values[i] for i in sorted_indices]
#     sorted_p_corr = [p_corrected[i] for i in sorted_indices]
#     sorted_sig = [is_significant[i] for i in sorted_indices]
    
#     # Create bar chart
#     bar_width = 0.35
#     x = np.arange(len(sorted_components))
    
#     # Original p-values
#     bars1 = plt.bar(x - bar_width/2, sorted_p, width=bar_width, label='Original p-value', 
#                    color='#4C72B0', edgecolor='black', linewidth=1)
    
#     # Corrected p-values
#     bars2 = plt.bar(x + bar_width/2, sorted_p_corr, width=bar_width, label='Holm-Bonferroni corrected', 
#                    color='#55A868', edgecolor='black', linewidth=1)
    
#     # Add alpha threshold line
#     plt.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    
#     # Add significance markers
#     for i, sig in enumerate(sorted_sig):
#         if sig:
#             plt.text(i, 0.001, '✓', ha='center', va='center', fontsize=16, 
#                    color='green', fontweight='bold')
#         else:
#             plt.text(i, 0.001, '✗', ha='center', va='center', fontsize=16, 
#                    color='red', fontweight='bold')
    
#     # Add numerical p-values on bars
#     for i, (p, p_corr) in enumerate(zip(sorted_p, sorted_p_corr)):
#         plt.text(x[i] - bar_width/2, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
#         plt.text(x[i] + bar_width/2, p_corr + 0.01, f'{p_corr:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
#     # Customize plot
#     plt.yscale('log')  # Log scale for better visibility
#     plt.ylabel('p-value (log scale)', fontsize=12)
#     plt.xlabel('Components (ordered by p-value)', fontsize=12)
#     plt.title('Holm-Bonferroni Correction for Multiple Comparisons', fontsize=15, pad=20)
#     plt.xticks(x, sorted_components, rotation=45, ha='right', fontsize=10)
#     plt.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # Add explanatory text
#     plt.figtext(0.5, 0.01, "Note: ✓ indicates significance maintained after correction (p < 0.05)", 
#                ha="center", fontsize=10, style='italic')
    
#     # Save figure
#     plt.tight_layout(rect=[0, 0.03, 1, 0.98])
#     plt.savefig('argumentation_holm_bonferroni_improved.png', dpi=300, bbox_inches='tight')
#     plt.savefig('argumentation_holm_bonferroni_improved.pdf', bbox_inches='tight')
#     plt.close()
    
#     # Text explanation for Holm-Bonferroni (printed to console)
#     print("\n=== HOLM-BONFERRONI CORRECTION EXPLANATION ===")
#     print("The Holm-Bonferroni method controls the familywise error rate when conducting multiple hypothesis tests.")
#     print("It works by ordering p-values from smallest to largest and applying sequential rejection:")
#     print()
#     print("Original significance threshold (α): 0.05")
#     print(f"Number of tests performed: {len(sorted_components)}")
#     print()
#     print("Procedure:")
#     for i, (component, p, p_corr, sig) in enumerate(zip(sorted_components, sorted_p, sorted_p_corr, sorted_sig)):
#         alpha_adjusted = 0.05 / (len(sorted_components) - i)
#         print(f"{i+1}. {component}: original p = {p:.6f}, adjusted α = {alpha_adjusted:.6f}, corrected p = {p_corr:.6f}")
#         if sig:
#             print("   ✓ Remains significant after correction")
#         else:
#             print("   ✗ No longer significant after correction")
            
#     print("\nCreated improved visualization of Holm-Bonferroni correction")

# if __name__ == "__main__":
#     results = analyze_argumentation_efficacy('pre_form.csv', 'post_form.csv')
        
#     # Extract data for Toulmin boxplot
#     pre_toulmin = results['composite_toulmin']['pre_data']
#     post_toulmin = results['composite_toulmin']['post_data']
    
#     # Create the combined density and boxplot visualization
#     create_toulmin_density_boxplot(pre_toulmin, post_toulmin)
    
#     # You can add other visualizations here as needed
#     print("Analysis complete - all visualizations generated.")











