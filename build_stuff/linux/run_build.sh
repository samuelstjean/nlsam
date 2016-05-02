source ~/buildfarm/bin/activate
cd ~/nlsam
pwd
python setup.py build_ext -i --force
pyinstaller build_stuff/linux/nlsam.spec --onefile
pyinstaller build_stuff/linux/stabilizer.spec --onefile
zip -j nlsam_linux_x64.zip dist/nlsam dist/stabilizer LICENSE README.md CHANGELOG.md
s
