# Cross Decoder

A package for pairwise decoding across multiple Latent Variable models.

## Installation

### Development Installation

To install the package in development mode (recommended for contributors):

```bash
# Clone the repository
git clone https://github.com/KabirDabholkar/cross_decoder.git
cd cross_decoder

# Install in editable mode
pip install -e .
```

### Regular Installation

To install the package directly from the source:

```bash
pip install .
```
## Usage
### Creating Your Own Analysis Class

To use the cross decoder with your model, create a class that inherits from `LatentAnalysisInterface`. The class needs to implement:

1. `get_latents(phase)` - Returns latent representations for your model for the specified phase ("train" or "val")
2. `get_trial_lengths(phase)` - Returns trial lengths for your model if applicable, otherwise None
3. `run_name` property - Returns a string identifier for your model

Example:
```
from cross_decoder import CrossDecoder, LatentAnalysisInterface

class YourAnalysis(LatentAnalysisInterface):
    def __init__(self,checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        
        self.model = load_model(checkpoint_path)
        self.dataset = load_dataset()

    def get_latents(self):
        latents = self.model(self.dataset)

    def run_name(self):
        return self.checkpoint_path
```



