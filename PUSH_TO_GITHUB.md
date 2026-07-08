# Push to GitHub

This repository is already initialized and committed locally.

1. Create an empty GitHub repository named `fwi-rtm-illumination-diagnostics`.
2. Do not add a README, license, or `.gitignore` on GitHub, because they already exist locally.
3. Push this local repository:

```powershell
git -C D:\Workspace\fwi-rtm-illumination-diagnostics remote add origin https://github.com/<your-account>/fwi-rtm-illumination-diagnostics.git
git -C D:\Workspace\fwi-rtm-illumination-diagnostics push -u origin main
```

If Git asks for authentication, use the GitHub browser authorization window or a GitHub personal access token.
