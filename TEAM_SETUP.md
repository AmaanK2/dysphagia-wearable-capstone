# Team setup

## First upload

Create a private GitHub repository for the team. If pushing this local folder as a new repository, leave GitHub's initial README, .gitignore and license options unselected; this folder already includes a README and .gitignore.

Unzip the source package and open the inner `swallow-detection` folder in VS Code. In that folder's terminal, replacing the URL below with your actual repository:

```bash
git init -b main
git add .
git commit -m "Add swallow detection baseline, CNN and validation"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

If the GitHub repository already has commits, clone it first and copy this package's contents into that clone instead. Then add, commit and push; do not force-push over existing work.

Add teammates as collaborators using the repository's access settings. They can then clone the repository and follow README.md.

## Teammate setup

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python -m venv .venv
```

Activate the environment for your operating system, then install the pinned dependencies and PyTorch as described in README.md. Check the installation:

```bash
python -m unittest test_pipeline.py test_cnn.py test_validation.py
```

Reading the Markdown reports and plots does not require installing Python packages or downloading the raw data. Training requires `signals.zip` and `labels.zip` in `data/raw/`; the download script and checksum checks are included. Prediction requires matching saved weights from the project owner or a local training run. Weights and datasets are intentionally absent from this source-only package.

## Everyday collaboration

Update main before starting work, then create a branch for one focused change:

```bash
git switch main
git pull
git switch -c feature/your-change
```

After implementing and testing your change:

```bash
git add .
git commit -m "Describe the change"
git push -u origin feature/your-change
```

Open a pull request on GitHub and have a teammate review it. Keep the purpose, changes and relevant test/experiment results in its description. Avoid modifying the same large script at the same time without coordinating.

## Project conventions

- Use a new experiment output directory for every run; keep baseline results unchanged.
- Keep participant groups separate. Use development data for model choices, not repeated pilot-test tuning.
- Describe scores as annotation-relative. Read label_review/README.md before treating unlabelled time as negative ground truth.
- Version reviewed annotations separately with reference evidence; do not silently edit original labels.
- Keep raw data, caches, virtual environments and model binaries out of commits. The .gitignore covers these file types.

The repository currently contains initial swallow-timing experiments. It does not include an abnormal-swallow or aspiration-risk classifier, or hardware integration.
