import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from functools import partial
import gc  # Add garbage collection import

def get_signal_r2_linear(
    signal_true_train, signal_pred_train, signal_true_val, signal_pred_val
):
    """
    Compute R2 score between two signals using linear regression.
    
    Args:
        signal_true_train: True signal for training (numpy array)
        signal_pred_train: Predicted signal for training (numpy array) 
        signal_true_val: True signal for validation (numpy array)
        signal_pred_val: Predicted signal for validation (numpy array)
        
    Returns:
        float: R2 score
    """
    # Function to compare the latent activity
    if len(signal_pred_train.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = signal_pred_train.shape
        signal_pred_train_flat = signal_pred_train.reshape(-1, n_d_pred)
        signal_pred_val_flat = signal_pred_val.reshape(-1, n_d_pred)
    else:
        signal_pred_train_flat = signal_pred_train
        signal_pred_val_flat = signal_pred_val

    if len(signal_true_train.shape) == 3:
        n_b_true, n_t_true, n_d_true = signal_true_train.shape
        signal_true_train_flat = signal_true_train.reshape(-1, n_d_true)
        signal_true_val_flat = signal_true_val.reshape(-1, n_d_true)
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
def compute_r2_for_pair_indices(i, j, latents_list_train, latents_list_val):
    """Helper function to compute R2 score for a pair of latents using indices."""
    # Reshape latents for linear regression and convert to numpy
    lats_i_train = latents_list_train[i].reshape(-1, latents_list_train[i].shape[-1])
    lats_j_train = latents_list_train[j].reshape(-1, latents_list_train[j].shape[-1])
    lats_i_val = latents_list_val[i].reshape(-1, latents_list_val[i].shape[-1])
    lats_j_val = latents_list_val[j].reshape(-1, latents_list_val[j].shape[-1])
    
    # Compute R2 score using train for fitting and val for evaluation
    return get_signal_r2_linear(
        signal_true_train=lats_i_train,
        signal_pred_train=lats_j_train,
        signal_true_val=lats_i_val,
        signal_pred_val=lats_j_val
    )

def compute_r2_for_pair_tensors(lats_i_train, lats_j_train, lats_i_val, lats_j_val):
    """Helper function to compute R2 score for a pair of latents using pre-indexed tensors."""
    # Reshape latents for linear regression and convert to numpy
    lats_i_train = lats_i_train.reshape(-1, lats_i_train.shape[-1])
    lats_j_train = lats_j_train.reshape(-1, lats_j_train.shape[-1])
    lats_i_val = lats_i_val.reshape(-1, lats_i_val.shape[-1])
    lats_j_val = lats_j_val.reshape(-1, lats_j_val.shape[-1])
    
    # Compute R2 score using train for fitting and val for evaluation
    return get_signal_r2_linear(
        signal_true_train=lats_i_train,
        signal_pred_train=lats_j_train,
        signal_true_val=lats_i_val,
        signal_pred_val=lats_j_val
    )

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

    def compute_pairwise_latent_r2(self, phase="val", parallel=False, n_jobs=None, max_trials_train=None, max_trials_val=None, use_indexed_tensors=False):
        """
        Compute pairwise R2 scores between latents of all analyses.

        Args:
            phase (str, optional): Phase to use for comparison ("train" or "val"). Defaults to "val".
            parallel (bool, optional): Whether to run comparisons in parallel. Defaults to False.
            n_jobs (int, optional): Number of parallel jobs. If None, uses all available CPUs. Defaults to None.
            max_trials_train (int, optional): Maximum number of trials to use for training. If None, uses all trials. Defaults to None.
            max_trials_val (int, optional): Maximum number of trials to use for validation. If None, uses all trials. Defaults to None.
            use_indexed_tensors (bool, optional): If True, passes indexed tensors to parallel processes. If False, passes indices and full lists. Defaults to False.

        Returns:
            tuple: (r2_matrix, group_matrix) where:
                - r2_matrix: Matrix of R2 scores between pairs of analyses
                - group_matrix: Matrix indicating which group each analysis belongs to
        """
        # Initialize matrices
        r2_matrix = np.zeros((self.num_analyses, self.num_analyses))
        group_matrix = np.zeros((self.num_analyses, self.num_analyses), dtype=object)

        # Get latents for all analyses for both train and val phases
        latents_list_train = []
        latents_list_val = []
        for analysis in self.analyses:
            # Get training latents
            latents_train = analysis.get_latents(phase="train").detach().numpy()
            trial_lens_train = analysis.get_trial_lengths(phase="train")
            if trial_lens_train is not None:
                latents_stack = []
                # Apply max_trials_train limit if specified
                if max_trials_train is not None:
                    trial_lens_train = trial_lens_train[:max_trials_train]
                    latents_train = latents_train[:max_trials_train]
                for i, t_len in enumerate(trial_lens_train):
                    latents_stack.append(latents_train[i, :int(t_len), :])
                latents_train = np.concatenate(latents_stack)
            latents_list_train.append(latents_train)

            # Get validation latents
            latents_val = analysis.get_latents(phase="val").detach().numpy()
            trial_lens_val = analysis.get_trial_lengths(phase="val")
            if trial_lens_val is not None:
                latents_stack = []
                # Apply max_trials_val limit if specified
                if max_trials_val is not None:
                    trial_lens_val = trial_lens_val[:max_trials_val]
                    latents_val = latents_val[:max_trials_val]
                for i, t_len in enumerate(trial_lens_val):
                    latents_stack.append(latents_val[i, :int(t_len), :])
                latents_val = np.concatenate(latents_stack)
            latents_list_val.append(latents_val)

        
        if parallel:
            # Create list of all pairs to process
            pairs = [(i, j) for i in range(self.num_analyses) for j in range(self.num_analyses)]
            
            # Set number of jobs
            if n_jobs is None:
                n_jobs = cpu_count()
            
            if use_indexed_tensors:
                # Process pairs in batches to reduce memory usage
                batch_size = 2*n_jobs  # Adjust based on available memory
                r2_values = []
                for batch_start in range(0, len(pairs), batch_size):
                    batch_end = min(batch_start + batch_size, len(pairs))
                    batch_pairs = pairs[batch_start:batch_end]
                    
                    # Create batch of tensor pairs
                    tensor_pairs = []
                    for i, j in batch_pairs:
                        tensor_pairs.append((
                            latents_list_train[i],
                            latents_list_train[j],
                            latents_list_val[i],
                            latents_list_val[j]
                        ))
                    
                    # Process batch
                    with Pool(n_jobs) as pool:
                        batch_r2_values = pool.starmap(compute_r2_for_pair_tensors, tensor_pairs)
                    r2_values.extend(batch_r2_values)
            else:
                # Create partial function with fixed latents lists
                compute_r2_partial = partial(
                    compute_r2_for_pair_indices,
                    latents_list_train=latents_list_train,
                    latents_list_val=latents_list_val
                )
                
                # Run parallel processing with indices
                with Pool(n_jobs) as pool:
                    r2_values = pool.starmap(compute_r2_partial, pairs)
            
            # Fill r2_matrix with results
            for idx, (i, j) in enumerate(pairs):
                r2_matrix[i, j] = r2_values[idx]
        else:
            # Original sequential processing
            for i in range(self.num_analyses):
                for j in range(self.num_analyses):
                    if use_indexed_tensors:
                        r2_matrix[i, j] = compute_r2_for_pair_tensors(
                            latents_list_train[i],
                            latents_list_train[j],
                            latents_list_val[i],
                            latents_list_val[j]
                        )
                    else:
                        r2_matrix[i, j] = compute_r2_for_pair_indices(
                            i, j,
                            latents_list_train=latents_list_train,
                            latents_list_val=latents_list_val
                        )

        # Fill group matrix
        for i in range(self.num_analyses):
            for j in range(self.num_analyses):
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