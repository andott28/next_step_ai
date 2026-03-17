#!/usr/bin/env python3
"""
Benchmark Comparison Analysis Script
Generates comprehensive comparison results from fixed benchmark data.
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BenchmarkComparisonAnalyzer:
    """
    Comprehensive benchmark comparison analyzer for model performance evaluation.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the comparison analyzer.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = results_dir
        self.comparison_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_benchmark_results(self, filename: str) -> Dict[str, Any]:
        """
        Load benchmark results from JSON file.
        
        Args:
            filename: Name of the results file
            
        Returns:
            Dictionary containing benchmark results
        """
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            raise ValueError(f"Error loading results file: {e}")
    
    def extract_model_performance(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract model performance data from benchmark results.
        
        Args:
            results: Raw benchmark results
            
        Returns:
            Structured model performance data
        """
        model_performance = {}
        
        if 'models' not in results:
            print("Warning: No 'models' key found in results")
            return model_performance
        
        for model_name, model_data in results['models'].items():
            performance = {
                'overall_score': model_data.get('overall_score', 0.0),
                'successful_evaluations': model_data.get('successful_evaluations', 0),
                'total_evaluations': model_data.get('total_evaluations', 0),
                'success_rate': model_data.get('successful_evaluations', 0) / max(1, model_data.get('total_evaluations', 1)),
                'statistics': model_data.get('statistics', {}),
                'task_statistics': model_data.get('task_statistics', {}),
                'run_scores': model_data.get('run_scores', [])
            }
            
            # Extract individual task scores
            task_scores = {}
            if 'glue_results' in model_data:
                for task_key, task_result in model_data['glue_results'].items():
                    if 'error' not in task_result and 'primary_score' in task_result:
                        # Extract task name from key (remove run suffix)
                        task_name = task_key.split('_run_')[0]
                        if task_name not in task_scores:
                            task_scores[task_name] = []
                        task_scores[task_name].append(task_result['primary_score'])
            
            # Calculate task statistics
            task_performance = {}
            for task_name, scores in task_scores.items():
                if scores:
                    task_performance[task_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores, ddof=1),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'ci_95': 1.96 * np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0,
                        'success_rate': len([s for s in scores if s > 0]) / len(scores)
                    }
            
            performance['task_performance'] = task_performance
            model_performance[model_name] = performance
        
        return model_performance
    
    def generate_overall_comparison(self, model_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overall comparison metrics between models.
        
        Args:
            model_performance: Structured model performance data
            
        Returns:
            Overall comparison metrics
        """
        comparison = {
            'model_count': len(model_performance),
            'models': list(model_performance.keys()),
            'rankings': [],
            'performance_gaps': {},
            'statistical_significance': {},
            'best_performing_tasks': {},
            'worst_performing_tasks': {}
        }
        
        # Create rankings
        model_scores = [(name, data['overall_score']) for name, data in model_performance.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model_name, score) in enumerate(model_scores, 1):
            comparison['rankings'].append({
                'rank': rank,
                'model_name': model_name,
                'score': score,
                'success_rate': model_performance[model_name]['success_rate']
            })
        
        # Calculate performance gaps
        if len(model_scores) >= 2:
            for i in range(len(model_scores) - 1):
                model1, score1 = model_scores[i]
                model2, score2 = model_scores[i + 1]
                
                if score1 > 0:
                    improvement_percent = ((score1 - score2) / score2) * 100
                else:
                    improvement_percent = 0
                
                comparison['performance_gaps'][f"{model1}_vs_{model2}"] = {
                    'model1': model1,
                    'model2': model2,
                    'model1_score': score1,
                    'model2_score': score2,
                    'absolute_difference': score1 - score2,
                    'improvement_percent': improvement_percent
                }
        
        # Analyze best and worst performing tasks
        all_tasks = set()
        for model_data in model_performance.values():
            all_tasks.update(model_data['task_performance'].keys())
        
        for task in all_tasks:
            task_scores = []
            for model_name, model_data in model_performance.items():
                if task in model_data['task_performance']:
                    task_scores.append((model_name, model_data['task_performance'][task]['mean']))
            
            if task_scores:
                task_scores.sort(key=lambda x: x[1], reverse=True)
                comparison['best_performing_tasks'][task] = {
                    'best_model': task_scores[0][0],
                    'best_score': task_scores[0][1],
                    'all_scores': task_scores
                }
                
                comparison['worst_performing_tasks'][task] = {
                    'worst_model': task_scores[-1][0],
                    'worst_score': task_scores[-1][1],
                    'all_scores': task_scores
                }
        
        return comparison
    
    def generate_category_analysis(self, model_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate category-specific analysis for different GLUE tasks.
        
        Args:
            model_performance: Structured model performance data
            
        Returns:
            Category-specific analysis
        """
        category_analysis = {
            'task_performance': {},
            'task_rankings': {},
            'task_difficulty': {},
            'model_strengths': {},
            'model_weaknesses': {}
        }
        
        # Get all unique tasks
        all_tasks = set()
        for model_data in model_performance.values():
            all_tasks.update(model_data['task_performance'].keys())
        
        # Analyze each task
        for task in all_tasks:
            task_scores = []
            for model_name, model_data in model_performance.items():
                if task in model_data['task_performance']:
                    score_data = model_data['task_performance'][task]
                    task_scores.append({
                        'model_name': model_name,
                        'mean': score_data['mean'],
                        'std': score_data['std'],
                        'ci_95': score_data['ci_95'],
                        'success_rate': score_data['success_rate']
                    })
            
            if task_scores:
                task_scores.sort(key=lambda x: x['mean'], reverse=True)
                category_analysis['task_performance'][task] = task_scores
                
                # Create task rankings
                category_analysis['task_rankings'][task] = [
                    {
                        'rank': i + 1,
                        'model_name': score['model_name'],
                        'score': score['mean'],
                        'confidence_interval': f"±{score['ci_95']:.4f}",
                        'success_rate': score['success_rate']
                    }
                    for i, score in enumerate(task_scores)
                ]
                
                # Calculate task difficulty (average performance across models)
                avg_performance = np.mean([score['mean'] for score in task_scores])
                std_performance = np.std([score['mean'] for score in task_scores])
                category_analysis['task_difficulty'][task] = {
                    'average_score': avg_performance,
                    'std_score': std_performance,
                    'difficulty_level': self._get_difficulty_level(avg_performance),
                    'consistency': 1.0 - (std_performance / avg_performance) if avg_performance > 0 else 1.0
                }
        
        # Identify model strengths and weaknesses
        for model_name, model_data in model_performance.items():
            strengths = []
            weaknesses = []
            
            for task, task_perf in model_data['task_performance'].items():
                # Compare model performance to task average
                task_avg = category_analysis['task_difficulty'][task]['average_score']
                model_score = task_perf['mean']
                
                if model_score > task_avg * 1.1:  # 10% above average
                    strengths.append({
                        'task': task,
                        'score': model_score,
                        'vs_average': model_score - task_avg
                    })
                elif model_score < task_avg * 0.9:  # 10% below average
                    weaknesses.append({
                        'task': task,
                        'score': model_score,
                        'vs_average': model_score - task_avg
                    })
            
            # Sort by performance difference
            strengths.sort(key=lambda x: x['vs_average'], reverse=True)
            weaknesses.sort(key=lambda x: x['vs_average'])
            
            category_analysis['model_strengths'][model_name] = strengths[:3]  # Top 3 strengths
            category_analysis['model_weaknesses'][model_name] = weaknesses[:3]  # Top 3 weaknesses
        
        return category_analysis
    
    def _get_difficulty_level(self, score: float) -> str:
        """Get difficulty level based on score."""
        if score >= 0.8:
            return "Easy"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.4:
            return "Hard"
        else:
            return "Very Hard"
    
    def generate_key_metrics(self, model_performance: Dict[str, Dict[str, Any]], 
                           comparison: Dict[str, Any], 
                           category_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate key performance metrics and insights.
        
        Args:
            model_performance: Structured model performance data
            comparison: Overall comparison metrics
            category_analysis: Category-specific analysis
            
        Returns:
            Key metrics and insights
        """
        key_metrics = {
            'overall_performance': {},
            'statistical_insights': {},
            'performance_highlights': {},
            'recommendations': []
        }
        
        # Overall performance metrics
        for model_name, model_data in model_performance.items():
            key_metrics['overall_performance'][model_name] = {
                'overall_score': model_data['overall_score'],
                'success_rate': model_data['success_rate'],
                'score_stability': model_data['statistics'].get('std', 0.0),
                'confidence_interval': f"±{model_data['statistics'].get('ci_95', 0.0):.4f}",
                'num_tasks_evaluated': len(model_data['task_performance'])
            }
        
        # Statistical insights
        if len(model_performance) >= 2:
            significant_differences = 0
            total_comparisons = 0
            
            for model1 in model_performance.keys():
                for model2 in model_performance.keys():
                    if model1 != model2:
                        total_comparisons += 1
                        # Check if performance difference is statistically significant
                        score1 = model_performance[model1]['overall_score']
                        score2 = model_performance[model2]['overall_score']
                        std1 = model_performance[model1]['statistics'].get('std', 0.0)
                        std2 = model_performance[model2]['statistics'].get('std', 0.0)
                        
                        # Simple significance check (could be improved with proper t-test)
                        if abs(score1 - score2) > (std1 + std2):
                            significant_differences += 1
            
            key_metrics['statistical_insights'] = {
                'significant_differences': significant_differences,
                'total_comparisons': total_comparisons,
                'significance_rate': significant_differences / total_comparisons if total_comparisons > 0 else 0,
                'most_stable_model': min(model_performance.keys(), 
                                       key=lambda x: model_performance[x]['statistics'].get('std', float('inf'))),
                'most_volatile_model': max(model_performance.keys(), 
                                         key=lambda x: model_performance[x]['statistics'].get('std', 0.0))
            }
        
        # Performance highlights
        if comparison['rankings']:
            best_model = comparison['rankings'][0]
            key_metrics['performance_highlights']['best_overall'] = {
                'model': best_model['model_name'],
                'score': best_model['score'],
                'achievement': f"Top performing model with {best_model['score']:.4f} score"
            }
        
        # Task-based highlights
        best_task_performance = {}
        for task, rankings in category_analysis['task_rankings'].items():
            if rankings:
                best_task_performance[task] = {
                    'best_model': rankings[0]['model_name'],
                    'best_score': rankings[0]['score'],
                    'difficulty': category_analysis['task_difficulty'][task]['difficulty_level']
                }
        
        key_metrics['performance_highlights']['best_task_performance'] = best_task_performance
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model_performance, comparison, category_analysis)
        key_metrics['recommendations'] = recommendations
        
        return key_metrics
    
    def _generate_recommendations(self, model_performance: Dict[str, Dict[str, Any]], 
                                comparison: Dict[str, Any], 
                                category_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Overall performance recommendations
        if comparison['rankings']:
            best_model = comparison['rankings'][0]['model_name']
            worst_model = comparison['rankings'][-1]['model_name']
            
            if best_model != worst_model:
                gap = comparison['rankings'][0]['score'] - comparison['rankings'][-1]['score']
                if gap > 0.1:  # Significant gap
                    recommendations.append(
                        f"Consider investigating why {best_model} outperforms {worst_model} by {gap:.4f} points"
                    )
        
        # Task-specific recommendations
        for task, difficulty in category_analysis['task_difficulty'].items():
            if difficulty['difficulty_level'] == "Very Hard":
                avg_score = difficulty['average_score']
                recommendations.append(
                    f"Task '{task}' is very hard (avg score: {avg_score:.4f}). Consider improving model architecture or training data."
                )
        
        # Model-specific recommendations
        for model_name, strengths in category_analysis['model_strengths'].items():
            if strengths:
                best_task = strengths[0]['task']
                recommendations.append(
                    f"Leverage {model_name}'s strength in {best_task} (score: {strengths[0]['score']:.4f}) for similar tasks"
                )
        
        for model_name, weaknesses in category_analysis['model_weaknesses'].items():
            if weaknesses:
                worst_task = weaknesses[0]['task']
                recommendations.append(
                    f"Improve {model_name}'s performance on {worst_task} (score: {weaknesses[0]['score']:.4f}) through targeted training"
                )
        
        return recommendations
    
    def generate_summary_report(self, comparison: Dict[str, Any],
                              category_analysis: Dict[str, Any],
                              key_metrics: Dict[str, Any]) -> str:
        """
        Generate a readable summary report of the findings.
        
        Args:
            comparison: Overall comparison metrics
            category_analysis: Category-specific analysis
            key_metrics: Key performance metrics
            
        Returns:
            Formatted summary report
        """
        report = []
        report.append("=" * 80)
        report.append("BENCHMARK COMPARISON ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models evaluated: {comparison['model_count']}")
        report.append("")
        
        # Overall rankings
        report.append("OVERALL PERFORMANCE RANKINGS")
        report.append("-" * 50)
        for ranking in comparison['rankings']:
            report.append(f"{ranking['rank']}. {ranking['model_name']}: {ranking['score']:.4f} "
                         f"(Success Rate: {ranking['success_rate']*100:.1f}%)")
        report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS")
        report.append("-" * 50)
        
        if 'statistical_insights' in key_metrics:
            insights = key_metrics['statistical_insights']
            report.append(f"• Statistical significance rate: {insights.get('significance_rate', 0)*100:.1f}%")
            report.append(f"• Most stable model: {insights.get('most_stable_model', 'N/A')}")
            report.append(f"• Most volatile model: {insights.get('most_volatile_model', 'N/A')}")
        report.append("")
        
        # Performance highlights
        if 'performance_highlights' in key_metrics:
            highlights = key_metrics['performance_highlights']
            if 'best_overall' in highlights:
                best = highlights['best_overall']
                report.append(f"Best overall performer: {best['model']} ({best['score']:.4f})")
            
            if 'best_task_performance' in highlights:
                report.append("Best task performance:")
                for task, perf in highlights['best_task_performance'].items():
                    report.append(f"   • {task}: {perf['best_model']} ({perf['best_score']:.4f}) [{perf['difficulty']}]")
        report.append("")
        
        # Task difficulty analysis
        report.append("TASK DIFFICULTY ANALYSIS")
        report.append("-" * 50)
        difficulty_counts = {"Easy": 0, "Medium": 0, "Hard": 0, "Very Hard": 0}
        for task, difficulty in category_analysis['task_difficulty'].items():
            difficulty_counts[difficulty['difficulty_level']] += 1
            report.append(f"• {task}: {difficulty['difficulty_level']} "
                         f"(avg: {difficulty['average_score']:.4f}, consistency: {difficulty['consistency']:.2f})")
        
        report.append("")
        report.append("Difficulty distribution:")
        for level, count in difficulty_counts.items():
            if count > 0:
                report.append(f"• {level}: {count} tasks")
        report.append("")
        
        # Recommendations
        if 'recommendations' in key_metrics and key_metrics['recommendations']:
            report.append("RECOMMENDATIONS")
            report.append("-" * 50)
            for i, rec in enumerate(key_metrics['recommendations'], 1):
                report.append(f"{i}. {rec}")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_comparison_results(self, comparison: Dict[str, Any],
                              category_analysis: Dict[str, Any],
                              key_metrics: Dict[str, Any],
                              summary_report: str) -> str:
        """
        Save comparison results to JSON file and summary report.
        
        Args:
            comparison: Overall comparison metrics
            category_analysis: Category-specific analysis
            key_metrics: Key performance metrics
            summary_report: Formatted summary report
            
        Returns:
            Path where results were saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results = {
            'analysis_timestamp': timestamp,
            'comparison': comparison,
            'category_analysis': category_analysis,
            'key_metrics': key_metrics,
            'summary_report': summary_report
        }
        
        results_file = f"results/benchmark_comparison_results_{timestamp}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        report_file = f"results/benchmark_comparison_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(summary_report)
        
        print(f"Results saved to:")
        print(f"   • {results_file}")
        print(f"   • {report_file}")
        
        return results_file
    
    def analyze_benchmark(self, results_file: str) -> Dict[str, Any]:
        """
        Complete benchmark analysis pipeline.
        
        Args:
            results_file: Name of the benchmark results file
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"Analyzing benchmark results from: {results_file}")
        
        # Load results
        results = self.load_benchmark_results(results_file)
        
        # Extract model performance
        model_performance = self.extract_model_performance(results)
        print(f"Extracted performance data for {len(model_performance)} models")
        
        # Generate overall comparison
        comparison = self.generate_overall_comparison(model_performance)
        print(f"Generated rankings for {len(comparison['rankings'])} models")
        
        # Generate category analysis
        category_analysis = self.generate_category_analysis(model_performance)
        print(f"Analyzed {len(category_analysis['task_performance'])} GLUE tasks")
        
        # Generate key metrics
        key_metrics = self.generate_key_metrics(model_performance, comparison, category_analysis)
        print(f"Generated {len(key_metrics.get('recommendations', []))} recommendations")
        
        # Generate summary report
        summary_report = self.generate_summary_report(comparison, category_analysis, key_metrics)
        
        # Save results
        results_file = self.save_comparison_results(comparison, category_analysis, key_metrics, summary_report)
        
        print(f"\nBenchmark analysis completed successfully!")
        print(f"Summary report saved to: {results_file.replace('.json', '.md')}")
        
        return {
            'results_file': results_file,
            'summary_report': summary_report,
            'comparison': comparison,
            'category_analysis': category_analysis,
            'key_metrics': key_metrics
        }


def main():
    """Main function to run benchmark comparison analysis."""
    print("BENCHMARK COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = BenchmarkComparisonAnalyzer()
    
    # Find the most recent GLUE evaluation results
    results_dir = "results"
    glue_files = [f for f in os.listdir(results_dir) if f.startswith('glue_evaluation_results_') and f.endswith('.json')]
    
    if not glue_files:
        print("No GLUE evaluation results found in results directory")
        return 1
    
    # Sort by timestamp (most recent first)
    glue_files.sort(reverse=True)
    latest_file = glue_files[0]
    
    print(f"Using latest results file: {latest_file}")
    
    try:
        # Run analysis
        analysis_results = analyzer.analyze_benchmark(latest_file)
        
        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(analysis_results['summary_report'])
        
        return 0
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())