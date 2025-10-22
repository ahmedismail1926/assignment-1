# 🚀 Quick Upload to GitHub - Command Reference

## One-Time Setup (Copy & Paste in Order)

```bash
# 1. Navigate to your project folder
cd "d:\fall2026\RL\assignment 1"

# 2. Initialize Git
git init

# 3. Configure Git (Replace with YOUR info)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 4. Add all files
git add .

# 5. Make first commit
git commit -m "Initial commit: Grid Maze Policy Iteration"
```

## Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `grid-maze-policy-iteration`
3. Make it Public or Private
4. **DON'T** check "Initialize with README"
5. Click "Create repository"

## Connect & Push (Replace YOUR_USERNAME)

```bash
# 6. Add remote (Replace YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/grid-maze-policy-iteration.git

# 7. Push to GitHub
git branch -M main
git push -u origin main
```

## When Asked for Password
- **DON'T** use your GitHub password
- **USE** Personal Access Token:
  1. Go to: https://github.com/settings/tokens
  2. Click "Generate new token (classic)"
  3. Select: `repo` scope
  4. Copy the token
  5. Paste as password

## 🎉 Done! Your Repository URL:
```
https://github.com/YOUR_USERNAME/grid-maze-policy-iteration
```

---

## Future Updates (After First Upload)

```bash
git add .
git commit -m "Your change description"
git push
```

## 📦 Files That Will Be Uploaded

✅ **Core Files:**
- environment.py
- policy_iteration.py  
- visualization.py
- main.py
- README.md
- requirements.txt
- LICENSE

✅ **Optional:**
- modified_test.py (legacy)
- videos/ (29 MP4 files)

❌ **Excluded:**
- __pycache__/
- .vscode/
- *.pyc files

## 🎬 Want to Exclude Videos?

Edit `.gitignore`, uncomment:
```
videos/
*.mp4
```

Then:
```bash
git add .gitignore
git commit -m "Exclude videos"
git push
```

---

**Need Help?** See `GITHUB_UPLOAD_GUIDE.md` for detailed instructions!
