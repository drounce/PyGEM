# build:
# python setup.py sdist bdist_wheel
poetry lock --no-update
poetry build

# upload:
python -m twine upload dist/*
# poetry publish  # was not able to publish with poetry for some reason...

# cleanup:
rm -r eggs/
rm -r dist/
rm -rf pygem.egg-info
rm -rf build/
