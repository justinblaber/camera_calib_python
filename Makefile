SRC = $(wildcard ./*.ipynb)

all: camera_calib_python docs

camera_calib_python: $(SRC)
	nbdev_build_lib
	touch camera_calib_python

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist