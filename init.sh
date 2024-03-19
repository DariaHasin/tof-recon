# This is a fix to not lose .so files when mounting docker volume
cp -r /tmp/nesvor/*.so /code/nesvor/

# Make the script hang
read