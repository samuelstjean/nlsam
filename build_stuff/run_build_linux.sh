source activate buildfarm
python setup.py install build_ext -i --force
pyinstaller nlsam_denoising.spec --onefile
zip -j nlsam_linux_x64.zip dist/nlsam_denoising LICENSE README.md CHANGELOG.md
zip -r nlsam_linux_x64.zip example/*
