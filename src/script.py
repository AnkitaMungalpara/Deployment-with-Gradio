import os
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig
import torch
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint if specified
    if cfg.get("ckpt_path"):
        log.info(f"Loading checkpoint: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    # Set model to eval mode
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 160, 160)  # Assuming standard image input size
    
    # Trace the model
    log.info("Tracing model...")
    traced_model = model.to_torchscript(method="trace", example_inputs=example_input)
    
    # Create output directory if it doesn't exist
    output_dir = Path("traced_models")
    output_dir.mkdir(exist_ok=True)
    
    # Save the traced model
    output_path = output_dir / "model.pt"
    torch.jit.save(traced_model, output_path)
    log.info(f"Traced model saved to: {output_path}")

if __name__ == "__main__":
    main()
