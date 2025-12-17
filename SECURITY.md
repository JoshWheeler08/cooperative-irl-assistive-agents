# Security Notice

## ⚠️ Important: API Key Exposure

**If you cloned this repository before December 17, 2025**, be aware that a WandB API key was previously exposed in the codebase. This key has been removed and should be considered compromised.

### Action Required

If you have this key in your git history:

1. **Revoke the old API key** in your WandB account settings
2. **Generate a new API key**
3. **Never commit API keys** to version control

## 🔐 Secure Configuration

### Quick Setup

```bash
# 1. Copy the example file
cp .env.example .env

# 2. Edit .env with your actual credentials
nano .env  # or use your preferred editor

# 3. Add your WandB API key (get from https://wandb.ai/authorize)
WANDB_API_KEY=your_actual_api_key_here
```

The `.env` file is already in `.gitignore` and will never be committed to version control.

### Setting Up API Keys

**Method 1: .env File (Recommended for Development)**

The project includes a `.env.example` template. Copy it and add your credentials:

```bash
cp .env.example .env
```

The code automatically loads environment variables from `.env` using `python-dotenv`.

**Method 2: System Environment Variable (Production/CI)**

```bash
# Add to your ~/.bashrc or ~/.zshrc
export WANDB_API_KEY="your_new_api_key_here"
```

**Method 3: Runtime Argument (Not Recommended)**

```bash
python3 experiments/run_main.py --wandb-api-key your_key_here
```

### Configuration Files

The configuration files (`code/src/configuration/*.yaml`) now have empty `API_KEY` fields. The code will automatically read from the `WANDB_API_KEY` environment variable.

## 🛡️ Security Best Practices

### What NOT to Commit

❌ **Never commit:**

- API keys or tokens
- Passwords or credentials
- Private keys
- Database connection strings with credentials
- OAuth tokens
- SSH keys

### Safe Practices

✅ **Always:**

- Use environment variables for sensitive data
- Add sensitive files to `.gitignore`
- Use secret management tools for production
- Rotate credentials if exposed
- Review commits before pushing

### Files Protected by .gitignore

The `.gitignore` now includes:

- `.env` and `.env.local`
- `secrets.yaml` and `*_secrets.yaml`

## 📝 Checking for Exposed Secrets

Before committing:

```bash
# Check for potential secrets
git diff | grep -E "(api_key|password|secret|token)" -i

# Review what you're committing
git diff --staged
```

## 🔍 If You Find a Security Issue

Please report security vulnerabilities by:

1. **DO NOT** open a public issue
2. Email the repository owner directly
3. Include details about the vulnerability
4. Wait for a response before public disclosure

## 📚 Additional Resources

- [WandB Security Best Practices](https://docs.wandb.ai/guides/technical-faq/security)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [Git Secret Management](https://git-secret.io/)

---

**Last Updated:** December 17, 2025
