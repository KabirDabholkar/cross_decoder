import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from abc import ABC, abstractmethod

def get_signal_r2_linear(
    signal_true_train, signal_pred_train, signal_true_val, signal_pred_val
):
    """
    Compute R2 score between two signals using linear regression.
    
    Args:
        signal_true_train: True signal for training
        signal_pred_train: Predicted signal for training
        signal_true_val: True signal for validation
        signal_pred_val: Predicted signal for validation
        
    Returns:
        float: R2 score
    """
    # Function to compare the latent activity
    if len(signal_pred_train.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = signal_pred_train.shape
        if isinstance(signal_pred_train, torch.Tensor):
            signal_pred_train_flat = (
                signal_pred_train.reshape(-1, n_d_pred).detach().numpy()
            )
            signal_pred_val_flat = signal_pred_val.reshape(-1, n_d_pred).detach().numpy()
        else:
            signal_pred_train_flat = signal_pred_train.reshape(-1, n_d_pred)
            signal_pred_val_flat = signal_pred_val.reshape(-1, n_d_pred)
    else:
        if isinstance(signal_pred_train, torch.Tensor):
            signal_pred_train_flat = signal_pred_train.detach().numpy()
            signal_pred_val_flat = signal_pred_val.detach().numpy()
        else:
            signal_pred_train_flat = signal_pred_train
            signal_pred_val_flat = signal_pred_val

    if len(signal_true_train.shape) == 3:
        n_b_true, n_t_true, n_d_true = signal_true_train.shape
        if isinstance(signal_true_train, torch.Tensor):
            signal_true_train_flat = (
                signal_true_train.reshape(-1, n_d_true).detach().numpy()
            )
            signal_true_val_flat = signal_true_val.reshape(-1, n_d_true).detach().numpy()
        else:
            signal_true_train_flat = signal_true_train.reshape(-1, n_d_true)
            signal_true_val_flat = signal_true_val.reshape(-1, n_d_true)
    else:
        if isinstance(signal_true_train, torch.Tensor):
            signal_true_train_flat = signal_true_train.detach().numpy()
            signal_true_val_flat = signal_true_val.detach().numpy()
        else:
            signal_true_train_flat = signal_true_train
            signal_true_val_flat = signal_true_val

    # Compare the latent activity
    reg = LinearRegression().fit(signal_true_train_flat, signal_pred_train_flat)
    preds = reg.predict(signal_true_val_flat)
    signal_r2_linear = r2_score(
        signal_pred_val_flat, preds, multioutput="variance_weighted"
    )
    return signal_r2_linear

class LatentAnalysisInterface(ABC):
    """
    Abstract base class defining the interface for latent analysis objects.
    Implement this interface for your specific framework.
    """
    @abstractmethod
    def get_latents(self, phase="val"):
        """
        Get latent representations.
        
        Args:
            phase (str): Phase to get latents for ("train" or "val")
            
        Returns:
            torch.Tensor: Latent representations
        """
        pass
    
    @abstractmethod
    def get_trial_lengths(self, phase="val"):
        """
        Get trial lengths if applicable.
        
        Args:
            phase (str): Phase to get trial lengths for ("train" or "val")
            
        Returns:
            torch.Tensor or None: Trial lengths if applicable, None otherwise
        """
        pass
    
    @property
    @abstractmethod
    def run_name(self):
        """Name of the analysis run"""
        pass

class CrossDecoder:
    """
    A class for evaluating pairwise latent comparisons between multiple analyses objects.
    """
    def __init__(self, comparison_tag=None):
        """
        Initialize the CrossDecoder.

        Args:
            comparison_tag (str, optional): Tag for identifying this comparison. Defaults to None.
        """
        self.comparison_tag = comparison_tag
        self.num_analyses = 0
        self.analyses = []
        self.groups = []

    def load_analysis(self, analysis, group="None"):
        """
        Load an analysis object into the CrossDecoder.

        Args:
            analysis: The analysis object to load (must implement LatentAnalysisInterface)
            group (str, optional): Group identifier for the analysis. Defaults to "None".
            
        Raises:
            TypeError: If analysis does not implement LatentAnalysisInterface
        """
        if not isinstance(analysis, LatentAnalysisInterface):
            raise TypeError("Analysis must implement LatentAnalysisInterface")
            
        self.analyses.append(analysis)
        self.groups.append(group)
        self.num_analyses += 1

    def regroup(self):
        """Sort analyses by group."""
        groups = np.array(self.groups)
        # Sort analyses by group
        sorted_inds = np.argsort(groups)
        self.analyses = [self.analyses[i] for i in sorted_inds]
        self.groups = groups[sorted_inds]

    def compute_pairwise_latent_r2(self, phase="val"):
        """
        Compute pairwise R2 scores between latents of all analyses.

        Args:
            phase (str, optional): Phase to use for comparison ("train" or "val"). Defaults to "val".

        Returns:
            tuple: (r2_matrix, group_matrix) where:
                - r2_matrix: Matrix of R2 scores between pairs of analyses
                - group_matrix: Matrix indicating which group each analysis belongs to
        """
        # Initialize matrices
        r2_matrix = np.zeros((self.num_analyses, self.num_analyses))
        group_matrix = np.zeros((self.num_analyses, self.num_analyses), dtype=object)

        # Get latents for all analyses
        latents_list = []
        for analysis in self.analyses:
            latents = analysis.get_latents(phase=phase).detach().numpy()
            
            # Handle unequal trial lengths if available
            trial_lens = analysis.get_trial_lengths(phase=phase)
            if trial_lens is not None:
                latents_stack = []
                for i, t_len in enumerate(trial_lens):
                    latents_stack.append(latents[i, :int(t_len), :])
                latents = np.concatenate(latents_stack)
            
            latents_list.append(latents)

        # Compute pairwise R2 scores
        for i in range(self.num_analyses):
            for j in range(self.num_analyses):
                if i == j:
                    r2_matrix[i, j] = 1.0  # Perfect correlation with self
                else:
                    # Reshape latents for linear regression
                    lats_i = latents_list[i].reshape(-1, latents_list[i].shape[-1])
                    lats_j = latents_list[j].reshape(-1, latents_list[j].shape[-1])
                    
                    # Compute R2 score
                    r2_matrix[i, j] = get_signal_r2_linear(
                        signal_true_train=lats_i,
                        signal_pred_train=lats_j,
                        signal_true_val=lats_i,
                        signal_pred_val=lats_j
                    )
                
                # Store group information
                group_matrix[i, j] = f"{self.groups[i]}-{self.groups[j]}"

        return r2_matrix, group_matrix

    def plot_pairwise_r2(self, r2_matrix, group_matrix=None, save_pdf=False):
        """
        Plot the pairwise R2 matrix.

        Args:
            r2_matrix: Matrix of R2 scores
            group_matrix: Matrix of group labels
            save_pdf (bool, optional): Whether to save the plot as PDF. Defaults to False.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(r2_matrix, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("R2 Score", rotation=-90, va="bottom")
        
        # Set labels
        ax.set_xticks(np.arange(self.num_analyses))
        ax.set_yticks(np.arange(self.num_analyses))
        ax.set_xticklabels([analysis.run_name for analysis in self.analyses])
        ax.set_yticklabels([analysis.run_name for analysis in self.analyses])
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        ax.set_title("Pairwise Latent R2 Scores")
        
        # Adjust layout
        fig.tight_layout()
        
        if save_pdf and self.comparison_tag:
            plt.savefig(f"{self.comparison_tag}_pairwise_r2.pdf")
            
        return fig

# Example implementation for ComputationThroughDynamics framework
class CTDAnalysisAdapter(LatentAnalysisInterface):
    """
    Adapter class for ComputationThroughDynamics analysis objects.
    """
    def __init__(self, analysis):
        self.analysis = analysis
        
    def get_latents(self, phase="val"):
        return self.analysis.get_latents(phase=phase)
    
    def get_trial_lengths(self, phase="val"):
        if hasattr(self.analysis, 'env') and self.analysis.env.dataset_name == "MultiTask":
            return self.analysis.get_trial_lens(phase=phase)
        return None
    
    @property
    def run_name(self):
        return self.analysis.run_name 