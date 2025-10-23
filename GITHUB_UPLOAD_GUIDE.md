# How to Upload This Project to GitHub

This guide will help you upload your Grid Maze Policy Iteration project to GitHub.

## Prerequisites

1. **Git installed** on your computer
   - Download from: https://git-scm.com/downloads
   - Verify installation: Open Command Prompt and type `git --version`

2. **GitHub account**
   - Create one at: https://github.com/signup

## Step-by-Step Guide

### 1. Initialize Git Repository (in your project folder)

Open Command Prompt in your project directory (`d:\fall2026\RL\assignment 1`) and run:

```bash
git init
```

### 2. Configure Git (First Time Only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Add Files to Git

```bash
git add .
```

This adds all files except those listed in `.gitignore`

### 4. Commit Your Changes

```bash
git commit -m "Initial commit: Grid Maze Policy Iteration project"
```

### 5. Create a New Repository on GitHub

1. Go to https://github.com/new
2. **Repository name**: `grid-maze-policy-iteration` (or your choice)
3. **Description**: "Policy Iteration algorithm for solving stochastic grid maze using Gymnasium"
4. **Visibility**: Choose Public or Private
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### 6. Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual values.

**Example:**
```bash
git remote add origin https://github.com/johndoe/grid-maze-policy-iteration.git
git branch -M main
git push -u origin main
```

### 7. Enter Credentials

When prompted, enter your GitHub credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your GitHub password)
  - Create token at: https://github.com/settings/tokens
  - Select scopes: `repo` (full control of private repositories)

## What Gets Uploaded?

✅ **Included:**
- `environment.py` - Grid Maze environment
- `policy_iteration.py` - Policy iteration algorithm
- `visualization.py` - Visualization utilities
- `main.py` - Main execution script
- `README.md` - Project documentation
- `modified_test.py` - Original implementation (legacy)
- `GITHUB_UPLOAD_GUIDE.md` - This guide
- `.gitignore` - Files to exclude

❌ **Excluded (by .gitignore):**
- `__pycache__/` - Python cache files
- `videos/` - **OPTIONAL**: Uncomment in `.gitignore` to exclude videos
- `.vscode/` - IDE settings
- Virtual environments

## Managing Videos

### Option A: Include Videos (Default)
Videos will be uploaded. This is fine if they're small (<25 files, <100MB total).

### Option B: Exclude Videos
1. Open `.gitignore`
2. Uncomment these lines:
   ```
   # videos/
   # *.mp4
   ```
   To:
   ```
   videos/
   *.mp4
   ```
3. Save and commit:
   ```bash
   git add .gitignore
   git commit -m "Exclude videos from repository"
   git push
   ```

### Option C: Use Git LFS for Large Videos
If videos are large (>100MB):
```bash
git lfs install
git lfs track "*.mp4"
git add .gitattributes
git commit -m "Track videos with Git LFS"
git push
```

## Future Updates

After making changes to your code:

```bash
git add .
git commit -m "Description of your changes"
git push
```

## Clone Repository Later

To download your project on another computer:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt  # Install dependencies
python main.py  # Run the project
```

## Troubleshooting

### Error: "Permission denied"
- Use a Personal Access Token instead of password
- Create at: https://github.com/settings/tokens

### Error: "Repository not found"
- Check the repository URL
- Verify repository visibility (public/private)
- Ensure you have access rights

### Error: "Large files detected"
- Use Git LFS for files >100MB
- Or exclude them via `.gitignore`

### Error: "Nothing to commit"
- Check if files are staged: `git status`
- Make sure `.gitignore` isn't excluding everything

## Quick Reference Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# View remote URL
git remote -v

# Pull latest changes
git pull

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main
```

## Additional Resources

- Git Documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com/
- GitHub Desktop (GUI): https://desktop.github.com/

---

**Need Help?** 
- GitHub Support: https://support.github.com/
- Git Tutorial: https://www.atlassian.com/git/tutorials
