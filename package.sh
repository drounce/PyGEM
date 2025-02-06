# build:
# python setup.py sdist bdist_wheel
poetry lock --no-update

# upload:
# python -m twine upload dist/*
poetry publish

# clean:
rm -r eggs/
rm -r dist/
rm -rf pygem.egg-info
rm -rf build/
