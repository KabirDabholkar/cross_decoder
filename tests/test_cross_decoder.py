import pytest
import torch
import numpy as np
from cross_decoder import CrossDecoder, LatentAnalysisInterface

class MockAnalysis(LatentAnalysisInterface):
    """Mock analysis class for testing with random latent data."""
    def __init__(self, name, latent_dim=10, num_trials=5, trial_length=100):
        self._name = name
        self.latent_dim = latent_dim
        self.num_trials = num_trials
        self.trial_length = trial_length
        
    def get_latents(self, phase="val"):
        # Generate random latent data
        return torch.randn(self.num_trials, self.trial_length, self.latent_dim)
    
    def get_trial_lengths(self, phase="val"):
        # For testing, we'll use fixed trial lengths
        return None
    
    @property
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
    
    # Check diagonal values (should be 1.0)
    assert r2_matrix[0, 0] == 1.0
    assert r2_matrix[1, 1] == 1.0
    
    # Check that off-diagonal values are between -1 and 1
    assert -1 <= r2_matrix[0, 1] <= 1
    assert -1 <= r2_matrix[1, 0] <= 1

def test_plot_pairwise_r2(cross_decoder, mock_analyses):
    """Test plotting R2 matrix."""
    # Load analyses
    cross_decoder.load_analysis(mock_analyses['analysis1'], group="group1")
    cross_decoder.load_analysis(mock_analyses['analysis2'], group="group1")
    
    # Compute R2 matrix
    r2_matrix, group_matrix = cross_decoder.compute_pairwise_latent_r2()
    
    # Test plotting
    fig = cross_decoder.plot_pairwise_r2(r2_matrix)
    assert fig is not None 