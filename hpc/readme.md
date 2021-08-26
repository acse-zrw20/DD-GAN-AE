* create_snapshots.sh will generate a set of structured meshes (subdomains) from a single domain in time on slug flow dataset
* reconstruct_to_vtu.sh does the inverse of the above, i.e. interpolates back to single unstructured mesh per timestep
* train_wandb_pred_sf.sh does predictive model hyperparameter optimization on slug flow dataset
* train_wandb_sf.sh does autoencoder model hyperparameter optimization on slug flow dataset