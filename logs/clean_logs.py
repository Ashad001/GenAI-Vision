import re
import os
from collections import defaultdict

def clean_gan_logs(input_file, output_file):
    """
    Clean GAN training logs from download.txt file.
    Extracts key information about GAN training for each class and overall results.
    
    Args:
        input_file (str): Path to the download.txt file
        output_file (str): Path to save the cleaned logs
    """
    print(f"Cleaning GAN logs from {input_file}...")
    
    # Patterns to match
    class_pattern = re.compile(r"--- Training GAN for class: (\w+) ---")
    epoch_pattern = re.compile(r"\[\d+/\d+\]\[\d+/\d+\] Loss_D: ([\d\.]+) Loss_G: ([\d\.]+) D\(x\): ([\d\.]+) D\(G\(z\)\): ([\d\.]+) / ([\d\.]+) Noise: ([\d\.]+)")
    best_loss_pattern = re.compile(r"New best G loss: ([\d\.]+)")
    finish_pattern = re.compile(r"Finished GAN training for (\w+) in ([\d\.]+) seconds")
    early_stopping_pattern = re.compile(r"Early stopping GAN training at epoch (\d+)")
    
    # Patterns for final results and discussion
    final_results_pattern = re.compile(r"===== Final Results =====")
    map_pattern = re.compile(r"Samples per class: (\d+), mAP: ([\d\.]+)")
    discussion_pattern = re.compile(r"===== Discussion =====")
    experiment_complete_pattern = re.compile(r"--- Experiment Complete ---")
    
    # Pattern for class distribution analysis
    class_distribution_pattern = re.compile(r"Class distribution analysis:")
    class_data_pattern = re.compile(r"(\w+)\s+(\d+)\s+(\d+)\s+([\d\.]+)")
    augmented_dataset_pattern = re.compile(r"Created augmented dataset with (\d+) original and (\d+) synthetic samples")
    
    # Data structures to store extracted information
    class_data = defaultdict(dict)
    current_class = None
    
    # For final results and discussion
    final_results = []
    discussion_lines = []
    capturing_discussion = False
    
    # For class distribution analysis
    class_distribution = []
    capturing_distribution = False
    augmented_dataset_info = None
    
    with open(input_file, 'r') as f:
        for line in f:
            # Remove timestamp at the beginning if present
            if ' ' in line and line.split(' ')[0].replace('.', '', 1).isdigit():
                line = ' '.join(line.split(' ')[1:])
            
            # Check for class distribution analysis section
            if class_distribution_pattern.search(line):
                capturing_distribution = True
                continue
            
            # Capture class distribution data
            if capturing_distribution:
                class_match = class_data_pattern.search(line)
                if class_match:
                    class_name, original, synthetic, ratio = class_match.groups()
                    class_distribution.append({
                        'class': class_name,
                        'original': int(original),
                        'synthetic': int(synthetic),
                        'ratio': float(ratio)
                    })
                
                # Check for augmented dataset info
                augmented_match = augmented_dataset_pattern.search(line)
                if augmented_match:
                    original, synthetic = augmented_match.groups()
                    augmented_dataset_info = {
                        'original': int(original),
                        'synthetic': int(synthetic)
                    }
                    capturing_distribution = False
                    continue
            
            # Check for final results section
            if final_results_pattern.search(line):
                capturing_discussion = False
                continue
                
            # Capture mAP values
            map_match = map_pattern.search(line)
            if map_match:
                samples, map_value = map_match.groups()
                final_results.append((int(samples), float(map_value)))
                continue
                
            # Check for discussion section
            if discussion_pattern.search(line):
                capturing_discussion = True
                continue
                
            # Check for experiment complete (end of discussion)
            if experiment_complete_pattern.search(line):
                capturing_discussion = False
                continue
                
            # Capture discussion lines
            if capturing_discussion and line.strip():
                # Clean the discussion line by removing timestamps and line numbers
                clean_line = line.strip()
                if ' ' in clean_line:
                    # Extract just the content, removing timestamp and line number
                    parts = clean_line.split(' ')
                    if len(parts) >= 2 and parts[0].replace('.', '', 1).isdigit() and parts[1].isdigit():
                        clean_line = ' '.join(parts[2:])
                discussion_lines.append(clean_line)
                continue
            
            # Check for class training start
            class_match = class_pattern.search(line)
            if class_match:
                current_class = class_match.group(1)
                class_data[current_class] = {
                    'epochs': [],
                    'best_loss': None,
                    'training_time': None,
                    'early_stopping_epoch': None
                }
                continue
            
            if current_class:
                # Check for epoch data
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    loss_d, loss_g, d_x, d_gz1, d_gz2, noise = map(float, epoch_match.groups())
                    class_data[current_class]['epochs'].append({
                        'loss_d': loss_d,
                        'loss_g': loss_g,
                        'd_x': d_x,
                        'd_gz1': d_gz1,
                        'd_gz2': d_gz2,
                        'noise': noise
                    })
                    continue
                
                # Check for best loss
                best_loss_match = best_loss_pattern.search(line)
                if best_loss_match:
                    class_data[current_class]['best_loss'] = float(best_loss_match.group(1))
                    continue
                
                # Check for training finish
                finish_match = finish_pattern.search(line)
                if finish_match:
                    class_name, training_time = finish_match.groups()
                    if class_name == current_class:
                        class_data[current_class]['training_time'] = float(training_time)
                        current_class = None
                    continue
                
                # Check for early stopping
                early_stopping_match = early_stopping_pattern.search(line)
                if early_stopping_match:
                    class_data[current_class]['early_stopping_epoch'] = int(early_stopping_match.group(1))
                    continue
    
    # Write cleaned data to output file
    with open(output_file, 'w') as f:
        f.write("GAN Training Summary\n")
        f.write("===================\n\n")
        
        # Add class distribution analysis section
        if class_distribution:
            f.write("Class Distribution Analysis\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Class':<15} {'Original':<10} {'Synthetic':<10} {'Ratio':<10}\n")
            f.write("-" * 50 + "\n")
            
            # Group by synthetic count to separate different distributions
            current_synthetic = None
            for item in class_distribution:
                if current_synthetic is not None and current_synthetic != item['synthetic']:
                    f.write("-" * 50 + "\n")  # Add separator line between different synthetic counts
                
                f.write(f"{item['class']:<15} {item['original']:<10} {item['synthetic']:<10} {item['ratio']:<10.2f}\n")
                current_synthetic = item['synthetic']
            
            if augmented_dataset_info:
                f.write("\n")
                f.write(f"Created augmented dataset with {augmented_dataset_info['original']} original and {augmented_dataset_info['synthetic']} synthetic samples\n")
            
            f.write("\n\n")
        
        for class_name, data in class_data.items():
            f.write(f"Class: {class_name}\n")
            f.write("-" * 50 + "\n")
            
            if data['best_loss'] is not None:
                f.write(f"Best Generator Loss: {data['best_loss']:.4f}\n")
            
            if data['training_time'] is not None:
                f.write(f"Training Time: {data['training_time']:.2f} seconds\n")
            
            if data['early_stopping_epoch'] is not None:
                f.write(f"Early Stopping at Epoch: {data['early_stopping_epoch']}\n")
            
            if data['epochs']:
                f.write(f"Total Epochs Recorded: {len(data['epochs'])}\n")
                
                # Calculate averages
                avg_loss_d = sum(epoch['loss_d'] for epoch in data['epochs']) / len(data['epochs'])
                avg_loss_g = sum(epoch['loss_g'] for epoch in data['epochs']) / len(data['epochs'])
                
                f.write(f"Average Discriminator Loss: {avg_loss_d:.4f}\n")
                f.write(f"Average Generator Loss: {avg_loss_g:.4f}\n")
            
            f.write("\n\n")
        
        # Add final results section
        if final_results:
            f.write("Final Results\n")
            f.write("-" * 50 + "\n")
            for samples, map_value in final_results:
                f.write(f"Samples per class: {samples}, mAP: {map_value:.4f}\n")
            f.write("\n\n")
        
        # Add discussion section
        if discussion_lines:
            f.write("Discussion\n")
            f.write("-" * 50 + "\n")
            
            # Filter out notebook conversion messages and other non-discussion content
            filtered_discussion = []
            for line in discussion_lines:
                # Skip lines with notebook conversion messages, warnings, etc.
                if any(term in line for term in ["NbConvertApp", "traitlets", "FutureWarning", "Converting notebook", "Writing", "warn("]):
                    continue
                # Skip empty lines that might have been left after cleaning
                if not line.strip():
                    continue
                filtered_discussion.append(line)
            
            # Write the cleaned discussion
            for line in filtered_discussion:
                f.write(f"{line}\n")
            
            # Add a blank line at the end
            f.write("\n")
    
    print(f"Cleaned logs saved to {output_file}")

if __name__ == "__main__":
    input_file = "logs/train_logs_raw.txt"
    output_file = "report/gan_training_summary.txt"
    
    if os.path.exists(input_file):
        clean_gan_logs(input_file, output_file)
    else:
        print(f"Error: Input file {input_file} not found.")
