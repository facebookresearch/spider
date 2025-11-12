# Git Guide for Documentation

## What to Commit ✅

### Documentation Source Files
- `docs/**/*.md` - All markdown files
- `docs/.vitepress/config.mts` - VitePress configuration
- `docs/package.json` - Dependencies list
- `docs/public/` - Static assets (images, etc.)
- `.github/workflows/deploy-docs.yml` - GitHub Actions workflow

### Root Files
- `CLAUDE.md` - Claude Code guidance
- `README.md` - Project README

## What NOT to Commit ❌

### Build Artifacts
- `docs/.vitepress/dist/` - Build output (auto-generated)
- `docs/.vitepress/cache/` - VitePress cache

### Dependencies
- `docs/node_modules/` - NPM packages (reinstall with `npm install`)
- `docs/package-lock.json` - Lock file (can be ignored or committed based on preference)

## Committing Your Changes

```bash
# Check what will be committed
git status

# Add documentation files
git add docs/ CLAUDE.md .github/workflows/deploy-docs.yml

# Commit
git commit -m "Add VitePress documentation"

# Push to trigger deployment
git push origin main
```

## Verifying .gitignore

Check that build artifacts are ignored:

```bash
# Should show as ignored
git check-ignore docs/node_modules
git check-ignore docs/.vitepress/dist
git check-ignore docs/.vitepress/cache

# Should NOT be ignored
git check-ignore docs/index.md
git check-ignore docs/package.json
```

## First Time Setup

```bash
# Clone repo
git clone https://github.com/yourusername/spider.git
cd spider

# Install docs dependencies
cd docs
npm install

# Start dev server
npm run docs:dev
```

## Deployment

Once pushed to GitHub:
1. Go to **Settings** → **Pages**
2. Set Source to **"GitHub Actions"**
3. GitHub Actions will automatically build and deploy
4. Visit `https://yourusername.github.io/spider/`

## File Size Tips

The `docs/public/figs/` directory contains images:
- These SHOULD be committed (source assets)
- If too large, consider using Git LFS
- Current size: Check with `du -sh docs/public/figs/`

## Troubleshooting

### Large Files Warning

If Git warns about large files:

```bash
# Check file sizes
find docs/public -type f -size +5M

# Consider Git LFS for large assets
git lfs track "docs/public/figs/*.gif"
git add .gitattributes
```

### Accidentally Committed node_modules

```bash
# Remove from git but keep locally
git rm -r --cached docs/node_modules
git commit -m "Remove node_modules from git"
```

### Clean Build Artifacts

```bash
# Remove all ignored files (be careful!)
cd docs
rm -rf node_modules .vitepress/dist .vitepress/cache
npm install
```
