import re
import sys
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np
import argparse
import os

def parse_miopen_log(log_path):
    results = []
    # if log_path does not exist
    if not os.path.exists(log_path):
        print(f"Log file {log_path} does not exist.")
        return results

    with open(log_path, 'r') as file:
        content = file.read()
        
    # Split the log file into individual test cases
    test_cases = re.split(r'MIOpenDriver convfp16', content)
    
    for test_case in test_cases[1:]:  # Skip the first empty element
        test_data = {}
        
        # Extract command
        command = "MIOpenDriver convfp16" + test_case.split('\n')[0]
        test_data['command'] = command
        
        # Extract solution
        solution_match = re.search(r'Solution: (\S+)', test_case)
        if solution_match:
            test_data['solution'] = solution_match.group(1)
        else:
            test_data['solution'] = "N/A"
        
        # Extract elapsed time
        elapsed_match = re.search(r'Elapsed: (\d+\.\d+)', test_case)
        if elapsed_match:
            test_data['elapsed'] = float(elapsed_match.group(1))
        else:
            test_data['elapsed'] = -1.0
        
        # Extract verification status and value
        verifies_match = re.search(r'Verifies OK.*?\((\d+\.\d+) < \d+\.\d+\)', test_case)
        failed_match = re.search(r'FAILED.*?\((\d+\.\d+) > \d+\.\d+\)', test_case)
        
        if verifies_match:
            test_data['verification'] = "OK"
            test_data['verify_value'] = float(verifies_match.group(1))
        elif failed_match:
            test_data['verification'] = "FAILED"
            test_data['verify_value'] = float(failed_match.group(1))
        else:
            test_data['verification'] = "N/A"
            test_data['verify_value'] = -1.0
        
        results.append(test_data)
    
    return results

def create_solution_summary(results):
    # Group by solution
    solution_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})
    
    for test in results:
        solution = test['solution']
        elapsed = test['elapsed']
        
        if elapsed > 0:
            solution_stats[solution]['count'] += 1
            solution_stats[solution]['total_time'] += elapsed
    
    # Calculate averages
    for solution in solution_stats:
        count = solution_stats[solution]['count']
        total_time = solution_stats[solution]['total_time']
        if count > 0:
            solution_stats[solution]['avg_time'] = total_time / count
        else:
            solution_stats[solution]['avg_time'] = 0
    
    return solution_stats

def plot_solution_performance(solution_stats):
    # Sort solutions by total time
    sorted_solutions = sorted(solution_stats.items(), 
                             key=lambda x: x[1]['total_time'], 
                             reverse=True)
    
    indices = list(range(1, len(sorted_solutions) + 1))
    total_times = [stats['total_time'] for _, stats in sorted_solutions]
    labels = [solution for solution, _ in sorted_solutions]
    
    plt.figure(figsize=(14, 8))  # Slightly larger figure for better spacing
    plt.plot(indices, total_times, marker='o', linestyle='')
    
    # Add rotated data labels
    for i, (idx, time, label) in enumerate(zip(indices, total_times, labels)):
        plt.annotate(f"{label}\n{time:.2f}ms", 
                     (idx, time),
                     xytext=(3, 3),  # Reduced vertical offset, added horizontal offset
                     textcoords="offset points", 
                     ha='left',
                     va='bottom',
                     rotation=15,
                     fontsize=9,
                     rotation_mode='anchor')  # Better rotation behavior
    
    plt.title('Total Execution Time by Solution')
    plt.xlabel('Solution Index')
    plt.ylabel('Total Time (ms)')
    plt.xticks(indices)  # Ensure all indices are shown
    plt.grid(True)
    
    # Adjust layout to prevent clipping of rotated labels
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add extra space at top
    
    # Save plot
    plt.savefig('solution_performance.png', bbox_inches='tight')  # Preserve labels
    print("Chart saved as 'solution_performance.png'")
    plt.show()
def plot_solution_comparison(solution_stats1, solution_stats2, label1="Log1", label2="Log2"):
    # Create combined solution list sorted by combined total time
    all_solutions = sorted(set(solution_stats1.keys()) | set(solution_stats2.keys()), 
                          key=lambda s: solution_stats1.get(s, {'total_time': 0})['total_time'] + 
                                       solution_stats2.get(s, {'total_time': 0})['total_time'], 
                          reverse=True)
    
    indices = np.arange(len(all_solutions))
    width = 0.4  # Width of each bar
    
    # Get total times for both logs
    total_times1 = [solution_stats1.get(sol, {'total_time': 0})['total_time'] for sol in all_solutions]
    total_times2 = [solution_stats2.get(sol, {'total_time': 0})['total_time'] for sol in all_solutions]
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Plot both datasets
    plt.bar(indices - width/2, total_times1, width, label=label1, color='skyblue', edgecolor='grey')
    plt.bar(indices + width/2, total_times2, width, label=label2, color='salmon', edgecolor='grey')
    
    # Add labels and title
    plt.title('Solution Performance Comparison', fontsize=16)
    plt.xlabel('Solutions', fontsize=12)
    plt.ylabel('Total Time (ms)', fontsize=12)
    plt.xticks(indices, all_solutions, rotation=10, ha='right', fontsize=10)
    plt.legend()
    
    # Add value labels on top of bars
    for i, (v1, v2) in enumerate(zip(total_times1, total_times2)):
        if v1 > 0:
            plt.text(i - width/2, v1 + 0.01*max(total_times1 + total_times2), 
                     f'{v1:.1f}', ha='center', va='bottom', fontsize=9)
        if v2 > 0:
            plt.text(i + width/2, v2 + 0.01*max(total_times1 + total_times2), 
                     f'{v2:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Add grid and adjust layout
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('solution_comparison.png', bbox_inches='tight', dpi=120)
    print("Comparison chart saved as 'solution_comparison.png'")
    plt.show()

def TestVerification(results):
    total_test = len(results)
    ok_count = sum(1 for test in results if test['verification'] == "OK")
    failed_count = sum(1 for test in results if test['verification'] == "FAILED")
    print(f"Total test count: {total_test}")
    print(f"Verification results: OK: {ok_count}, Failed: {failed_count}")
    print()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare MIOpen performance logs')
    parser.add_argument('--log1', required=True, help='First MIOpen log file')
    parser.add_argument('--log2', required=True, help='Second MIOpen log file')
    args = parser.parse_args()
    
    log_path1 = args.log1
    log_path2 = args.log2
    results1 = parse_miopen_log(log_path1)
    results2 = parse_miopen_log(log_path2)
    
    # Print basic results
    print("LOG1:")
    TestVerification(results1)
    print("LOG2:")
    TestVerification(results2)
    
    # Generate solution summary
    solution_stats1  = create_solution_summary(results1)
    solution_stats2 = create_solution_summary(results2)

    # Create combined solution list
    all_solutions = sorted(set(solution_stats1.keys()) | set(solution_stats2.keys()), 
                          key=lambda s: solution_stats1.get(s, {'total_time': 0})['total_time'] + 
                                       solution_stats2.get(s, {'total_time': 0})['total_time'], 
                          reverse=True)

    # Print combined solution summary
    # Print combined solution summary
    print("\nSolution Summary:")
    print(f"IDX | {'Count1':>6} | {'Total1(ms)':>12} | {'Avg1(ms)':>10} | {'Count2':>6} | {'Total2(ms)':>12} | {'Avg2(ms)':>10} | {'Gap (%)':>10} | Solution")
    print('-' * 125)

    for idx, solution in enumerate(all_solutions, 1):
        stats1 = solution_stats1.get(solution, {'count': 0, 'total_time': 0, 'avg_time': 0})
        stats2 = solution_stats2.get(solution, {'count': 0, 'total_time': 0, 'avg_time': 0})
    
        count1 = stats1['count']
        total_time1 = stats1['total_time']
        avg_time1 = stats1['avg_time']
    
        count2 = stats2['count']
        total_time2 = stats2['total_time']
        avg_time2 = stats2['avg_time']
    
        # Calculate gap percentage
        if count1 > 0 and count2 > 0:
            gap_percent = ((avg_time1 - avg_time2) / avg_time1) * 100
            gap_str = f"{gap_percent:+.1f}%"
        elif count1 == 0 and count2 > 0:
            gap_str = "N/A"  # Solution is new in log2
        elif count2 == 0 and count1 > 0:
            gap_str = "N/A"  # Solution is missing in log2
        else:
            gap_str = "N/A"  # Solution not present in either log

        print(f"{idx:3d} | {count1:6d} | {total_time1:12.4f} | {avg_time1:10.4f} | "
              f"{count2:6d} | {total_time2:12.4f} | {avg_time2:10.4f} | "
              f"{gap_str:>10} | {solution}")

    # Add summary row
    total_count1 = sum(stats['count'] for stats in solution_stats1.values())
    total_time1 = sum(stats['total_time'] for stats in solution_stats1.values())
    avg_time1 = total_time1 / total_count1 if total_count1 > 0 else 0

    total_count2 = sum(stats['count'] for stats in solution_stats2.values())
    total_time2 = sum(stats['total_time'] for stats in solution_stats2.values())
    avg_time2 = total_time2 / total_count2 if total_count2 > 0 else 0

    # Calculate overall gap percentage
    if total_count1 > 0 and total_count2 > 0:
        total_gap = ((avg_time2 - avg_time1) / avg_time1) * 100
        total_gap_str = f"{total_gap:+.1f}%"
    else:
        total_gap_str = "N/A"

    print('-' * 125)
    print(f"{'TOTAL':>5} | {total_count1:6d} | {total_time1:12.4f} | {avg_time1:10.4f} | "
          f"{total_count2:6d} | {total_time2:12.4f} | {avg_time2:10.4f} | "
          f"{total_gap_str:>10} |")
    # Create performance chart
    # plot_solution_performance(solution_stats1)
    plot_solution_comparison(solution_stats1=solution_stats1, solution_stats2=solution_stats2)
    
    # Optionally save to CSV
    save_to_csv = input("Save detailed results1 to CSV? (y/n): ").lower() == 'y'
    if save_to_csv:
        csv_path = log_path1 + ".csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['command', 'solution', 'elapsed', 'verification', 'verify_value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for test in results1:
                writer.writerow(test)
        
        print(f"Detailed results1 saved to {csv_path}")

if __name__ == "__main__":
    main()
