import pytest
import torch
import numpy as np
from cross_decoder import CrossDecoder, LatentAnalysisInterface, get_signal_r2_linear
import os
import json
import hashlib
from datetime import datetime

class MockAnalysis(LatentAnalysisInterface):
    """Mock analysis class for testing with random latent data."""
    def __init__(self, name, latent_dim=10, num_trials=5, trial_length=100, base_latents=None, seed=42):
        self._name = name
        self.latent_dim = latent_dim
        self.num_trials = num_trials
        self.trial_length = trial_length
        self._base_latents = base_latents
        self.seed = seed
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        
        # Pre-generate latents
        if self._base_latents is not None:
            # If we have base latents, extend them with random values
            base_dim = self._base_latents.shape[-1]
            self._latents = torch.zeros(self.num_trials, self.trial_length, self.latent_dim)
            self._latents[:, :, :base_dim] = self._base_latents
            # Use the same random state for consistent results
            self._latents[:, :, base_dim:] = torch.from_numpy(
                self.rng.randn(self.num_trials, self.trial_length, self.latent_dim - base_dim)
            )
        else:
            # Generate random latent data using the same random state
            self._latents = torch.from_numpy(
                self.rng.randn(self.num_trials, self.trial_length, self.latent_dim)
            )
        
    def get_latents(self, phase="val"):
        return self._latents
    
    def get_trial_lengths(self, phase="val"):
        # For testing, we'll use fixed trial lengths
        return None
    
    def run_name(self):
        return self._name

@pytest.fixture
def mock_analyses():
    """Fixture providing mock analyses for testing."""
    return {
        'analysis1': MockAnalysis("analysis1", latent_dim=10),
        'analysis2': MockAnalysis("analysis2", latent_dim=10),
        'analysis3': MockAnalysis("analysis3", latent_dim=15)  # Different dimension
    }

@pytest.fixture
def cross_decoder():
    """Fixture providing a CrossDecoder instance."""
    return CrossDecoder(comparison_tag="test")

def test_load_analysis(cross_decoder, mock_analyses):
    """Test loading analyses into CrossDecoder."""
    # Test loading single analysis
    cross_decoder.load_analysis(mock_analyses['analysis1'], group="group1")
    assert cross_decoder.num_analyses == 1
    
    # Test loading multiple analyses
    cross_decoder.load_analysis(mock_analyses['analysis2'], group="group1")
    cross_decoder.load_analysis(mock_analyses['analysis3'], group="group2")
    assert cross_decoder.num_analyses == 3

def test_regroup(cross_decoder, mock_analyses):
    """Test regrouping analyses."""
    # Load analyses in mixed order
    cross_decoder.load_analysis(mock_analyses['analysis2'], group="group2")
    cross_decoder.load_analysis(mock_analyses['analysis1'], group="group1")
    cross_decoder.load_analysis(mock_analyses['analysis3'], group="group1")
    
    # Regroup
    cross_decoder.regroup()
    
    # Check that groups are sorted
    assert cross_decoder.groups[0] == "group1"
    assert cross_decoder.groups[1] == "group1"
    assert cross_decoder.groups[2] == "group2"

def test_compute_pairwise_latent_r2(cross_decoder, mock_analyses):
    """Test computing pairwise R2 scores."""
    # Load analyses
    cross_decoder.load_analysis(mock_analyses['analysis1'], group="group1")
    cross_decoder.load_analysis(mock_analyses['analysis2'], group="group1")
    
    # Compute R2 matrix
    r2_matrix, group_matrix = cross_decoder.compute_pairwise_latent_r2()
    
    # Check matrix shape
    assert r2_matrix.shape == (2, 2)
    
    # Check diagonal values (should be 1.0) with more lenient tolerance
    np.testing.assert_array_almost_equal(r2_matrix[0, 0], 1.0, decimal=4)
    np.testing.assert_array_almost_equal(r2_matrix[1, 1], 1.0, decimal=4)
    
    # Check that off-diagonal values are between -1 and 1
    assert -1 <= r2_matrix[0, 1] <= 1
    assert -1 <= r2_matrix[1, 0] <= 1

def test_plot_pairwise_r2(cross_decoder, mock_analyses):
    """Test plotting R2 matrix."""
    # Create test_plots directory if it doesn't exist
    os.makedirs('test_plots', exist_ok=True)
    
    # Load analyses
    cross_decoder.load_analysis(mock_analyses['analysis1'], group="group1")
    cross_decoder.load_analysis(mock_analyses['analysis2'], group="group1")
    
    # Compute R2 matrix
    r2_matrix, group_matrix = cross_decoder.compute_pairwise_latent_r2()
    
    # Test plotting
    fig = cross_decoder.plot_pairwise_r2(r2_matrix)
    fig.savefig('test_plots/cross_decoding_test.png')
    assert fig is not None

def create_increasing_dim_analyses(base_dim=5, num_trials=30, trial_length=10, latent_dims=None, base_seed=42):
    """Create a series of mock analyses with increasing dimensions sharing base latents.
    
    Args:
        base_dim (int): Dimension of the base latent space
        num_trials (int): Number of trials per analysis
        trial_length (int): Length of each trial
        latent_dims (list): List of dimensions for each analysis. If None, uses [5, 8, 12, 15, 20]
        base_seed (int): Seed for the base analysis. Each subsequent analysis gets seed+1
        
    Returns:
        tuple: (base_latents, analyses) where:
            - base_latents: The base latent space tensor
            - analyses: List of MockAnalysis objects with increasing dimensions
    """
    if latent_dims is None:
        latent_dims = [5, 8, 12, 15, 20]
        
    # Create first analysis with base latent space
    base_analysis = MockAnalysis("base_analysis", latent_dim=base_dim, 
                               num_trials=num_trials, trial_length=trial_length,
                               seed=base_seed)
    base_latents = base_analysis.get_latents()
    
    # Create series of analyses with increasing dimensions
    analyses = []
    for i, dim in enumerate(latent_dims):
        analysis = MockAnalysis(
            f"analysis_{dim}d", 
            latent_dim=dim,
            num_trials=num_trials, 
            trial_length=trial_length,
            base_latents=base_latents,
            seed=base_seed + i + 1  # Each analysis gets a different seed
        )
        analyses.append(analysis)
        
    return base_latents, analyses

def test_shared_base_latents_series(cross_decoder):
    """Test cross-decoding between analyses with shared base latents and increasing dimensions."""
    # Create test_plots directory if it doesn't exist
    os.makedirs('test_plots', exist_ok=True)
    
    # Create analyses with increasing dimensions
    _, analyses = create_increasing_dim_analyses()
    
    # Load analyses into cross decoder
    for analysis in analyses:
        cross_decoder.load_analysis(analysis, group=f"group_{analysis.latent_dim}d")
    
    # Compute R2 matrix
    r2_matrix, _ = cross_decoder.compute_pairwise_latent_r2()
    
    # Save results
    cross_decoder.save_dir = 'test_plots'
    cross_decoder.save_decoding_matrices(r2_matrix, None)
    
    # Load results
    loaded_r2, _, metadata = cross_decoder.load_decoding_matrices("decoding")
    print(metadata)
    
    # Verify loaded data matches original
    print(r2_matrix, loaded_r2)
    np.testing.assert_allclose(r2_matrix, loaded_r2, rtol=1e-4)
    
    # Verify matrix properties
    assert r2_matrix.shape == (5, 5)  # 5x5 matrix for 5 analyses
    
    # Plot the matrix
    fig = cross_decoder.plot_pairwise_r2(1-r2_matrix)
    
    # Save the plot in test_plots directory
    plot_path = 'test_plots/shared_base_latents_matrix.png'
    fig.savefig(plot_path)
    assert os.path.exists(plot_path)
    
    # Print the matrix for inspection
    print("\nR2 Matrix for analyses with shared base latents:")
    print("Dimensions:", [a.latent_dim for a in analyses])
    print(r2_matrix)

def test_parallel_pairwise_latent_r2(cross_decoder):
    """Test parallel computation of pairwise R2 scores with increasing dimensions."""
    # Create test_plots directory if it doesn't exist
    os.makedirs('test_plots', exist_ok=True)
    
    # Create analyses with increasing dimensions
    _, analyses = create_increasing_dim_analyses()
    
    # Load analyses into cross decoder
    for analysis in analyses:
        cross_decoder.load_analysis(analysis, group=f"group_{analysis.latent_dim}d")
    
    # Compute R2 matrices using both sequential and parallel methods
    r2_matrix_seq, _ = cross_decoder.compute_pairwise_latent_r2(parallel=False)
    r2_matrix_par, _ = cross_decoder.compute_pairwise_latent_r2(parallel=True)
    
    # Check that matrices have correct shape
    assert r2_matrix_seq.shape == (5, 5)
    assert r2_matrix_par.shape == (5, 5)
    
    # Check that results are identical between sequential and parallel
    np.testing.assert_array_almost_equal(r2_matrix_seq, r2_matrix_par, decimal=4)
    
    # Save results
    cross_decoder.save_dir = 'test_plots'
    cross_decoder.save_decoding_matrices(r2_matrix_par, None)
    
    # Plot the matrix
    fig = cross_decoder.plot_pairwise_r2(1-r2_matrix_par)
    
    # Save the plot
    plot_path = 'test_plots/parallel_shared_base_latents_matrix.png'
    fig.savefig(plot_path)
    assert os.path.exists(plot_path)
    
    # Print the matrix for inspection
    print("\nR2 Matrix for parallel computation:")
    print("Dimensions:", [a.latent_dim for a in analyses])
    print(r2_matrix_par)

def test_related_latent_spaces():
    """Test creating mock analyses with related latent spaces."""
    # Create base latent space
    base_dim = 5
    num_trials = 3
    trial_length = 10
    
    # Create first analysis with base latent space
    analysis1 = MockAnalysis("base_analysis", latent_dim=base_dim, 
                           num_trials=num_trials, trial_length=trial_length)
    
    # Get the base latents
    base_latents = analysis1.get_latents()
    
    # Create second analysis with extended latent space using the base latents
    extended_dim = 8  # 5 base dimensions + 3 additional random dimensions
    analysis2 = MockAnalysis("extended_analysis", latent_dim=extended_dim,
                           num_trials=num_trials, trial_length=trial_length,
                           base_latents=base_latents)
    
    # Get the extended latents
    extended_latents = analysis2.get_latents()
    
    # Verify dimensions
    assert base_latents.shape == (num_trials, trial_length, base_dim)
    assert extended_latents.shape == (num_trials, trial_length, extended_dim)
    
    # Verify that the first base_dim dimensions of extended_latents match base_latents
    np.testing.assert_array_almost_equal(
        extended_latents[:, :, :base_dim].numpy(),
        base_latents.numpy(),
        decimal=5
    )
    
    # Verify that the additional dimensions are random
    additional_dims = extended_latents[:, :, base_dim:]
    assert additional_dims.shape == (num_trials, trial_length, extended_dim - base_dim)
    # Check that these dimensions are not all zeros
    assert not torch.allclose(additional_dims, torch.zeros_like(additional_dims))

def test_save_load_plot_decoding_matrices(cross_decoder, mock_analyses, tmp_path):
    """Test saving, loading, and plotting decoding matrices with IDs."""
    print("mock_analyses['analysis1'].run_name():", mock_analyses['analysis1'].run_name())
    # Load analyses
    cross_decoder.load_analysis(mock_analyses['analysis1'], group="group1")
    cross_decoder.load_analysis(mock_analyses['analysis2'], group="group1")
    
    # Compute R2 matrix
    r2_matrix, _ = cross_decoder.compute_pairwise_latent_r2()
    
    # Save matrices
    cross_decoder.save_dir = str(tmp_path)  # Use tmp_path for saving
    cross_decoder.save_decoding_matrices(r2_matrix, None)  # Don't save group matrix
    
    # Load matrices
    loaded_r2, _, metadata = cross_decoder.load_decoding_matrices("decoding")
    
    # Verify loaded data matches original
    np.testing.assert_array_almost_equal(r2_matrix, loaded_r2, decimal=4)
    
    # Verify metadata contains expected information
    assert "analysis_ids" in metadata
    assert "analysis_names" in metadata
    assert "groups" in metadata
    assert len(metadata["analysis_ids"]) == 2
    assert len(metadata["analysis_names"]) == 2
    
    # Test plotting with IDs from metadata
    fig = cross_decoder.plot_pairwise_r2(
        loaded_r2,
    )
    
    # Save the plot
    plot_path = tmp_path / "decoding_matrix_plot.png"
    fig.savefig(plot_path)
    assert plot_path.exists()
    
    # Verify plot has correct number of ticks
    assert len(fig.axes[0].get_xticklabels()) == 2
    assert len(fig.axes[0].get_yticklabels()) == 2

def test_get_signal_r2_linear_deterministic():
    """Test that get_signal_r2_linear gives the same result for the same input data."""
    
    # Generate random test data
    num_trials = 3
    trial_length = 10
    latent_dim = 5
    
    # Create fixed random data using a seed
    np.random.seed(42)
    signal_true_train = np.random.randn(num_trials, trial_length, latent_dim)
    signal_pred_train = np.random.randn(num_trials, trial_length, latent_dim)
    signal_true_val = np.random.randn(num_trials, trial_length, latent_dim)
    signal_pred_val = np.random.randn(num_trials, trial_length, latent_dim)
    
    # Compute R2 score twice
    r2_score1 = get_signal_r2_linear(
        signal_true_train, signal_pred_train,
        signal_true_val, signal_pred_val
    )
    r2_score2 = get_signal_r2_linear(
        signal_true_train, signal_pred_train,
        signal_true_val, signal_pred_val
    )
    
    # Verify results are identical
    np.testing.assert_almost_equal(r2_score1, r2_score2)
    
    # Also test with flattened data
    signal_true_train_flat = signal_true_train.reshape(-1, latent_dim)
    signal_pred_train_flat = signal_pred_train.reshape(-1, latent_dim)
    signal_true_val_flat = signal_true_val.reshape(-1, latent_dim)
    signal_pred_val_flat = signal_pred_val.reshape(-1, latent_dim)
    
    r2_score3 = get_signal_r2_linear(
        signal_true_train_flat, signal_pred_train_flat,
        signal_true_val_flat, signal_pred_val_flat
    )
    
    # Verify results are identical for both 3D and 2D inputs
    np.testing.assert_almost_equal(r2_score1, r2_score3)

def save_decoding_matrices(self, r2_matrix, group_matrix, phase="val", save_json_only=False):
    """
    Save decoding matrices with metadata.

    Args:
        r2_matrix (np.ndarray): Matrix of R2 scores
        group_matrix (np.ndarray): Matrix of group labels
        phase (str, optional): Phase used for decoding. Defaults to "val"
        save_json_only (bool, optional): If True, only save the metadata JSON file. Defaults to False
    """
    # Generate deterministic hash IDs for each analysis
    analysis_ids = []
    analysis_names = []
    for analysis in self.analyses:
        # Get run name properly handling both property and method cases
        if isinstance(analysis.run_name, property):
            run_name = analysis.run_name.fget(analysis)
        else:
            run_name = analysis.run_name()
        analysis_names.append(run_name)
        # Create deterministic hash from run name
        analysis_id = hashlib.md5(run_name.encode()).hexdigest()[:8]
        analysis_ids.append(analysis_id)
    
    # Create metadata dictionary
    metadata = {
        "analysis_ids": analysis_ids,
        "analysis_names": analysis_names,
        "groups": self.groups,
        "phase": phase,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save metadata
    metadata_path = os.path.join(self.save_dir, f"{phase}_decoding_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if not save_json_only:
        # Save matrices
        np.save(os.path.join(self.save_dir, f"{phase}_r2_matrix.npy"), r2_matrix)
        if group_matrix is not None:
            np.save(os.path.join(self.save_dir, f"{phase}_group_matrix.npy"), group_matrix)