"""
Model Evaluation Script
Analyzes pose classification performance:
- Confusion matrix
- Class distribution
- Confidence scores
- Per-class metrics

Usage:
  python evaluate_model.py predictions_log.json --ground-truth ground_truth.json
  python evaluate_model.py predictions_log.json --interactive
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys


class ModelEvaluator:
    """Evaluate pose classification model performance"""
    
    def __init__(self, predictions_file, ground_truth_file=None):
        self.predictions_file = predictions_file
        self.ground_truth_file = ground_truth_file
        
        # Load predictions
        print(f"Loading predictions from: {predictions_file}")
        with open(predictions_file) as f:
            data = json.load(f)
        
        self.predictions = data['predictions']
        self.metadata = data['metadata']
        
        print(f"  Total predictions: {len(self.predictions)}")
        print(f"  Duration: {self.metadata['duration']:.2f}s")
        print(f"  Kalman enabled: {self.metadata['kalman_enabled']}")
        
        # Load ground truth if provided
        self.ground_truth = None
        if ground_truth_file:
            print(f"\nLoading ground truth from: {ground_truth_file}")
            with open(ground_truth_file) as f:
                self.ground_truth = json.load(f)
            print(f"  Ground truth samples: {len(self.ground_truth)}")
    
    def get_class_distribution(self):
        """Get distribution of predicted classes"""
        poses = [p['pose'] for p in self.predictions]
        return Counter(poses)
    
    def get_confidence_stats(self):
        """Get confidence statistics per class"""
        stats = {}
        
        # Group by pose
        pose_confidences = {}
        for pred in self.predictions:
            pose = pred['pose']
            conf = pred['confidence']
            
            if pose not in pose_confidences:
                pose_confidences[pose] = []
            pose_confidences[pose].append(conf)
        
        # Calculate statistics
        for pose, confidences in pose_confidences.items():
            stats[pose] = {
                'count': len(confidences),
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'median': np.median(confidences)
            }
        
        return stats
    
    def align_with_ground_truth(self):
        """Align predictions with ground truth timestamps"""
        if not self.ground_truth:
            return None, None
        
        y_true = []
        y_pred = []
        
        for gt_sample in self.ground_truth:
            gt_time = gt_sample['timestamp']
            gt_label = gt_sample['true_pose']
            
            # Find closest prediction
            closest_pred = min(
                self.predictions,
                key=lambda p: abs(p['timestamp'] - gt_time)
            )
            
            # Only include if within 0.5 seconds
            if abs(closest_pred['timestamp'] - gt_time) < 0.5:
                y_true.append(gt_label)
                y_pred.append(closest_pred['pose'])
        
        return y_true, y_pred
    
    def plot_class_distribution(self, save_path=None):
        """Plot class distribution bar chart"""
        dist = self.get_class_distribution()
        
        plt.figure(figsize=(10, 6))
        poses = list(dist.keys())
        counts = list(dist.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(poses)))
        bars = plt.bar(poses, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Pose Class', fontsize=14)
        plt.ylabel('Number of Predictions', fontsize=14)
        plt.title('Class Distribution of Pose Predictions', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Class distribution saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confidence_distribution(self, save_path=None):
        """Plot confidence score distributions per class"""
        stats = self.get_confidence_stats()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        poses = list(stats.keys())
        confidences_by_pose = {}
        for pred in self.predictions:
            pose = pred['pose']
            if pose not in confidences_by_pose:
                confidences_by_pose[pose] = []
            confidences_by_pose[pose].append(pred['confidence'])
        
        axes[0].boxplot([confidences_by_pose[p] for p in poses], labels=poses)
        axes[0].set_xlabel('Pose Class', fontsize=12)
        axes[0].set_ylabel('Confidence Score', fontsize=12)
        axes[0].set_title('Confidence Distribution by Class (Box Plot)', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(poses, rotation=45, ha='right')
        
        # Bar plot with error bars
        means = [stats[p]['mean'] for p in poses]
        stds = [stats[p]['std'] for p in poses]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(poses)))
        bars = axes[1].bar(poses, means, yerr=stds, color=colors, alpha=0.8, 
                           capsize=5, edgecolor='black')
        
        axes[1].set_xlabel('Pose Class', fontsize=12)
        axes[1].set_ylabel('Mean Confidence ± Std', fontsize=12)
        axes[1].set_title('Mean Confidence by Class', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(poses, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confidence distribution saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix if ground truth available"""
        if not self.ground_truth:
            print("⚠ Ground truth not provided, skipping confusion matrix")
            return
        
        y_true, y_pred = self.align_with_ground_truth()
        
        if not y_true:
            print("⚠ No aligned predictions found")
            return
        
        # Get unique classes
        classes = sorted(list(set(y_true + y_pred)))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Proportion'},
                   linewidths=0.5, linecolor='gray')
        
        plt.xlabel('Predicted Pose', fontsize=14)
        plt.ylabel('True Pose', fontsize=14)
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, labels=classes))
        
        # Print accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    def plot_confidence_timeline(self, save_path=None):
        """Plot confidence scores over time"""
        timestamps = [p['timestamp'] for p in self.predictions]
        confidences = [p['confidence'] for p in self.predictions]
        poses = [p['pose'] for p in self.predictions]
        
        # Normalize timestamps to start at 0
        timestamps = np.array(timestamps) - timestamps[0]
        
        plt.figure(figsize=(14, 6))
        
        # Color by pose
        unique_poses = sorted(list(set(poses)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_poses)))
        pose_colors = {pose: colors[i] for i, pose in enumerate(unique_poses)}
        
        for pose in unique_poses:
            mask = np.array(poses) == pose
            plt.scatter(np.array(timestamps)[mask], np.array(confidences)[mask],
                       label=pose, color=pose_colors[pose], alpha=0.6, s=20)
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.title('Confidence Scores Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confidence timeline saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Class distribution
        dist = self.get_class_distribution()
        print("\nClass Distribution:")
        for pose, count in dist.most_common():
            percentage = (count / len(self.predictions)) * 100
            print(f"  {pose:15s}: {count:5d} ({percentage:5.1f}%)")
        
        # Confidence statistics
        stats = self.get_confidence_stats()
        print("\nConfidence Statistics:")
        for pose, stat in stats.items():
            print(f"\n  {pose}:")
            print(f"    Count:   {stat['count']}")
            print(f"    Mean:    {stat['mean']:.3f}")
            print(f"    Std:     {stat['std']:.3f}")
            print(f"    Min:     {stat['min']:.3f}")
            print(f"    Max:     {stat['max']:.3f}")
            print(f"    Median:  {stat['median']:.3f}")
        
        # Overall statistics
        all_confidences = [p['confidence'] for p in self.predictions]
        print("\n" + "-"*60)
        print(f"Overall Mean Confidence: {np.mean(all_confidences):.3f}")
        print(f"Overall Std Confidence:  {np.std(all_confidences):.3f}")
        print(f"Total Predictions:       {len(self.predictions)}")
        print(f"Unknown Predictions:     {dist.get('unknown', 0)} ({(dist.get('unknown', 0)/len(self.predictions)*100):.1f}%)")
    
    def run_full_evaluation(self, output_dir="evaluation_results"):
        """Run complete evaluation and save all plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print("RUNNING FULL EVALUATION")
        print(f"{'='*60}")
        
        # Print summary
        self.print_summary()
        
        # Generate plots
        print(f"\nGenerating plots...")
        self.plot_class_distribution(output_dir / "class_distribution.png")
        self.plot_confidence_distribution(output_dir / "confidence_distribution.png")
        self.plot_confidence_timeline(output_dir / "confidence_timeline.png")
        
        if self.ground_truth:
            self.plot_confusion_matrix(output_dir / "confusion_matrix.png")
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")


def create_ground_truth_template():
    """Create template for ground truth annotation"""
    template = {
        "metadata": {
            "description": "Ground truth labels for pose predictions",
            "instructions": "Add timestamp and true_pose for each sample"
        },
        "samples": [
            {
                "timestamp": 1234567890.123,
                "true_pose": "fist"
            },
            {
                "timestamp": 1234567891.234,
                "true_pose": "flat_hand"
            }
        ]
    }
    
    with open("ground_truth_template.json", 'w') as f:
        json.dump(template, f, indent=2)
    
    print("✓ Ground truth template created: ground_truth_template.json")


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python evaluate_model.py <predictions_log.json> [options]")
        print("\nOptions:")
        print("  --ground-truth <file>  : Path to ground truth labels file")
        print("  --output-dir <dir>     : Directory for saving plots (default: evaluation_results)")
        print("  --create-template      : Create ground truth template file")
        print("\nExamples:")
        print("  python evaluate_model.py predictions_log.json")
        print("  python evaluate_model.py predictions_log.json --ground-truth labels.json")
        print("  python evaluate_model.py predictions_log.json --output-dir results")
        print("  python evaluate_model.py --create-template")
        sys.exit(1)
    
    # Create template mode
    if '--create-template' in sys.argv:
        create_ground_truth_template()
        return
    
    predictions_file = sys.argv[1]
    
    # Parse options
    ground_truth_file = None
    output_dir = "results"
    
    if '--ground-truth' in sys.argv:
        idx = sys.argv.index('--ground-truth')
        ground_truth_file = sys.argv[idx + 1]
    
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        output_dir = sys.argv[idx + 1]
    
    # Run evaluation
    evaluator = ModelEvaluator(predictions_file, ground_truth_file)
    evaluator.run_full_evaluation(output_dir)


if __name__ == "__main__":
    main()