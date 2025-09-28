# Install Poppler for PDF Processing

The Gawk application needs Poppler to convert PDF pages to images for analysis. Here's how to install it:

## Windows Installation

### Option 1: Download Pre-built Binary (Recommended)
1. Go to: https://github.com/oschwartz10612/poppler-windows/releases/
2. Download the latest `poppler-xx.xx.x_x64.7z` file
3. Extract it to a folder like `C:\poppler`
4. Add `C:\poppler\bin` to your system PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click "Environment Variables"
   - Under "System Variables", find and select "Path", click "Edit"
   - Click "New" and add `C:\poppler\bin`
   - Click "OK" on all dialogs
5. Restart your command prompt/PowerShell

### Option 2: Using Conda (if you have Anaconda/Miniconda)
```bash
conda install -c conda-forge poppler
```

### Option 3: Using Chocolatey (if you have it installed)
```bash
choco install poppler
```

## Linux Installation

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

### CentOS/RHEL:
```bash
sudo yum install poppler-utils
```

### Fedora:
```bash
sudo dnf install poppler-utils
```

## macOS Installation

### Using Homebrew:
```bash
brew install poppler
```

### Using MacPorts:
```bash
sudo port install poppler
```

## Verify Installation

After installing, verify it works by running:
```bash
python test_gawk.py
```

Or test directly:
```bash
pdftoppm -h
```

## What Happens Without Poppler

If Poppler is not installed, the Gawk application will:
- Still work for recording videos
- Create placeholder PDF page images for analysis
- Show a warning in the logs
- Analysis will work but with limited PDF processing

The application is designed to be robust and will continue working even without Poppler, just with reduced PDF functionality.
