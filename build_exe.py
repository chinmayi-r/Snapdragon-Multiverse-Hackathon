"""
Build script for creating Windows executable from Gawk application.
"""

import os
import sys
import subprocess
import shutil

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("PyInstaller already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def create_spec_file():
    """Create PyInstaller spec file for the application"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gawk_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('gawk_data', 'gawk_data'),
    ],
    hiddenimports=[
        'cv2',
        'mediapipe',
        'numpy',
        'PIL',
        'PyPDF2',
        'pdf2image',
        'matplotlib',
        'seaborn',
        'sklearn',
        'pandas',
        'scipy',
        'mss',
        'pyautogui',
        'tkinter',
        'threading',
        'json',
        'os',
        'time',
        'datetime'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Gawk',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''
    
    with open('gawk.spec', 'w') as f:
        f.write(spec_content)
    
    print("Created gawk.spec file")

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable...")
    
    # Create the executable
    cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',
        '--name=Gawk',
        '--add-data=gawk_data;gawk_data',
        '--hidden-import=cv2',
        '--hidden-import=mediapipe',
        '--hidden-import=numpy',
        '--hidden-import=PIL',
        '--hidden-import=PyPDF2',
        '--hidden-import=pdf2image',
        '--hidden-import=matplotlib',
        '--hidden-import=seaborn',
        '--hidden-import=sklearn',
        '--hidden-import=pandas',
        '--hidden-import=scipy',
        '--hidden-import=mss',
        '--hidden-import=pyautogui',
        'gawk_launcher.py'
    ]
    
    try:
        subprocess.check_call(cmd)
        print("Executable built successfully!")
        print("Find the executable in the 'dist' folder")
    except subprocess.CalledProcessError as e:
        print(f"Error building executable: {e}")
        return False
    
    return True

def create_installer():
    """Create a simple installer script"""
    installer_content = '''@echo off
echo Installing Gawk Attention Analysis System...
echo.

REM Create application directory
if not exist "C:\\Program Files\\Gawk" mkdir "C:\\Program Files\\Gawk"

REM Copy executable
copy "dist\\Gawk.exe" "C:\\Program Files\\Gawk\\"

REM Create desktop shortcut
echo [InternetShortcut] > "%USERPROFILE%\\Desktop\\Gawk.lnk"
echo URL=file:///C:/Program Files/Gawk/Gawk.exe >> "%USERPROFILE%\\Desktop\\Gawk.lnk"

echo.
echo Installation complete!
echo You can now run Gawk from the desktop shortcut or from:
echo C:\\Program Files\\Gawk\\Gawk.exe
echo.
pause
'''
    
    with open('install.bat', 'w') as f:
        f.write(installer_content)
    
    print("Created installer script: install.bat")

def main():
    """Main build process"""
    print("Gawk Windows Executable Builder")
    print("==============================")
    
    # Check if we're on Windows
    if os.name != 'nt':
        print("This build script is designed for Windows only.")
        return
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    if build_executable():
        # Create installer
        create_installer()
        
        print("\nBuild process completed!")
        print("Files created:")
        print("- dist/Gawk.exe (main executable)")
        print("- install.bat (installer script)")
        print("\nTo install:")
        print("1. Run install.bat as administrator")
        print("2. Or manually copy dist/Gawk.exe to desired location")
    else:
        print("Build failed!")

if __name__ == "__main__":
    main()
