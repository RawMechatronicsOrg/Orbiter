"""Object-mask generation for COLMAP (keep what rotates with the table,
drop the static room).

Modules import as flat siblings — generate_colmap_masks.py puts this folder on
sys.path so ``import mask_pipeline`` etc. resolve whether the package is
imported or the CLI is launched by absolute path / symlink.
"""
